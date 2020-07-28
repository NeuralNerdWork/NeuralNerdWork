package neuralnerdwork.backprop;

import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

import java.util.Arrays;
import java.util.stream.IntStream;

import static java.lang.String.format;
import static neuralnerdwork.math.VectorSum.sum;

public record ConvolutionLayer(int inputChannels, Convolution[] convolutions, ActivationFunction activation) implements Layer<ConvolutionLayer.ConvolutionCache> {

    public ConvolutionLayer {
        record Lengths(int input, int output) {}
        long uniqueLengths = Arrays.stream(convolutions)
                                   .map(c -> new Lengths(c.inputLength(), c.outputLength()))
                                   .distinct()
                                   .count();
        if (uniqueLengths > 1) {
            throw new IllegalArgumentException("convolutions must all have same input and output sizes");
        }
        if (convolutions.length == 0) {
            throw new IllegalArgumentException("convolutions array must be non-empty");
        }
    }

    @Override
    public boolean containsVariable(int variable) {
        return Arrays.stream(convolutions)
                     .anyMatch(c -> c.matrix().filter().containsVariable(variable) || c.bias().variable() == variable);
    }

    @Override
    public int outputLength() {
        return convolutions[0].outputLength() * convolutions.length * inputChannels;
    }

    @Override
    public int inputLength() {
        return convolutions[0].inputLength() * convolutions.length * inputChannels;
    }

    @Override
    public IntStream variables() {
        return Arrays.stream(convolutions)
                     .flatMapToInt(c -> IntStream.concat(c.matrix.filter().variables(),
                                                         IntStream.of(c.bias.variable()))
                     );
    }

    @Override
    public Result<Vector, ConvolutionCache> evaluate(Vector layerInput, Model.ParameterBindings bindings) {
        if (layerInput.length() != inputLength()) {
            throw new IllegalArgumentException(format("given vector length [%d] does not match expected input size [%d]", layerInput
                    .length(), inputLength()));
        }

        double[] values = new double[outputLength()];
        ChannelCache[][] channels = new ChannelCache[convolutions.length][inputChannels];

        for (int convIndex = 0; convIndex < convolutions.length; convIndex++) {
            for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                try {
                    ConvolutionFilterMatrix matrix = convolutions[convIndex].matrix();
                    ScalarParameter bias = convolutions[convIndex].bias();
                    Vector activationInput = sum(
                            new MatrixVectorProduct(
                                    matrix,
                                    layerInput
                            ),
                            new RepeatedScalarVectorExpression(bias, matrix.rows())
                    ).evaluate(bindings);
                    Vector activation = new VectorizedSingleVariableFunction(this.activation, activationInput).evaluate(bindings);
                    DMatrix activationWithRespectConvolution = new DiagonalizedVector(
                            new VectorizedSingleVariableFunction(
                                    this.activation,
                                    activationInput
                            )
                    ).evaluate(bindings);
                    channels[convIndex][channelIndex] = new ChannelCache(activation, activationInput, activationWithRespectConvolution);
                    System.arraycopy(activation.toArray(),
                                     0,
                                     values,
                                     (convIndex * inputChannels + channelIndex) * convolutions[0].outputLength(),
                                     activation.length());
                } catch (RuntimeException ex) {
                    throw new RuntimeException(format("encountered exception at (convIndex, channelIndex)=(%d, %d)", convIndex, channelIndex), ex);
                }
            }
        }

        ConstantVector combinedActivation = new ConstantVector(values);
        return new Result<>(combinedActivation, new ConvolutionCache(combinedActivation, channels));
    }

    @Override
    public Result<DMatrix, ConvolutionCache> derivativeWithRespectToLayerInput(Vector layerInput, ConvolutionCache cache, Model.ParameterBindings bindings) {
        DMatrixRMaj derivative = new DMatrixRMaj(outputLength(), inputLength());
        for (int convIndex = 0; convIndex < convolutions.length; convIndex++) {
            for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                ChannelCache cacheEntry = cache.channels()[convIndex][channelIndex];
                DMatrix activationWithRespectConvolution = cacheEntry.activationWithRespectConvolution();
                ConvolutionFilterMatrix matrix = convolutions[convIndex].matrix();
                DMatrix convolutionDerivative = new MatrixProduct(new DMatrixExpression(activationWithRespectConvolution), matrix).evaluate(bindings);
                int rowOffset = (convIndex * inputChannels + channelIndex) * convolutions[0].outputLength();
                int colOffset = (convIndex * inputChannels + channelIndex) * convolutions[0].inputLength();
                for (int row = 0; row < convolutionDerivative.getNumRows(); row++) {
                    for (int col = 0; col < convolutionDerivative.getNumCols(); col++) {
                        derivative.set(rowOffset + row, colOffset + col, convolutionDerivative.get(row, col));
                    }
                }
            }
        }

        return new Result<>(derivative, cache);
    }

    @Override
    public Result<Vector, ConvolutionCache> derivativeWithRespectLayerParameter(Vector layerInput, int variable, ConvolutionCache cache, Model.ParameterBindings bindings) {
        double[] derivative = new double[outputLength()];
        for (int convIndex = 0; convIndex < convolutions.length; convIndex++) {
            for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                ChannelCache cacheEntry = cache.channels()[convIndex][channelIndex];
                DMatrix activationWithRespectConvolution = cacheEntry.activationWithRespectConvolution();
                ConvolutionFilterMatrix matrix = convolutions[convIndex].matrix();
                Vector convolutionDerivative =
                        new MatrixVectorProduct(
                                new MatrixProduct(
                                        new DMatrixExpression(activationWithRespectConvolution),
                                        new DMatrixExpression(matrix.computePartialDerivative(bindings, variable))
                                ),
                                layerInput
                        ).evaluate(bindings);
                double[] derivativeValues = convolutionDerivative.toArray();
                System.arraycopy(derivativeValues,
                                 0,
                                 derivative,
                                 (convIndex * inputChannels + channelIndex) * convolutions[0].inputLength(),
                                 convolutionDerivative.length());
            }
        }

        return new Result<>(new ConstantVector(derivative), cache);
    }

    @Override
    public Vector getEvaluation(ConvolutionCache cache) {
        return cache.output();
    }

    public record Convolution(ConvolutionFilterMatrix matrix, ScalarParameter bias) {
        public int inputLength() {
            return matrix.cols();
        }

        public int outputLength() {
            return matrix.rows();
        }
    }

    public record ConvolutionCache(Vector output, ChannelCache[][] channels) {}

    public record ChannelCache(Vector activation, Vector activationInput, DMatrix activationWithRespectConvolution) {}
}
