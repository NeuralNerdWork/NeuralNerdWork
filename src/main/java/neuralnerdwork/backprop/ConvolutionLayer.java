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
    public Result<DMatrix, ConvolutionCache> evaluate(DMatrix layerInput, Model.ParameterBindings bindings) {
        if (layerInput.getNumRows() != inputLength()) {
            throw new IllegalArgumentException(format("given vector length [%d] does not match expected input size [%d]", layerInput
                    .getNumRows(), inputLength()));
        }

        double[] values = new double[outputLength()];
        ChannelCache[][] channels = new ChannelCache[convolutions.length][inputChannels];

        for (int convIndex = 0; convIndex < convolutions.length; convIndex++) {
            for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                try {
                    ConvolutionFilterMatrix matrix = convolutions[convIndex].matrix();
                    ScalarParameter bias = convolutions[convIndex].bias();
                    DMatrix activationInput = sum(
                            new MatrixVectorProduct(
                                    matrix,
                                    new DMatrixColumnVectorExpression(layerInput)
                            ),
                            new RepeatedScalarVectorExpression(bias, matrix.rows())
                    ).evaluate(bindings);
                    DMatrixColumnVectorExpression activationInputExpression = new DMatrixColumnVectorExpression(activationInput);
                    DMatrix activation = new VectorizedSingleVariableFunction(this.activation, activationInputExpression).evaluate(bindings);
                    DMatrix activationWithRespectConvolution = new DiagonalizedVector(
                            new VectorizedSingleVariableFunction(
                                    this.activation,
                                    activationInputExpression
                            )
                    ).evaluate(bindings);
                    channels[convIndex][channelIndex] = new ChannelCache(activation, activationInput, activationWithRespectConvolution);
                    int dstOffset = (convIndex * inputChannels + channelIndex) * convolutions[0].outputLength();
                    for (int i = 0; i < activation.getNumRows(); i++) {
                        values[dstOffset + i] = activation.get(i, 0);
                    }
                } catch (RuntimeException ex) {
                    throw new RuntimeException(format("encountered exception at (convIndex, channelIndex)=(%d, %d)", convIndex, channelIndex), ex);
                }
            }
        }

        DMatrixRMaj combinedActivation = new DMatrixRMaj(values.length, 1, true, values);
        return new Result<>(combinedActivation, new ConvolutionCache(combinedActivation, channels));
    }

    @Override
    public Result<DMatrix, ConvolutionCache> derivativeWithRespectToLayerInput(DMatrix layerInput, ConvolutionCache cache, Model.ParameterBindings bindings) {
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
    public Result<DMatrix, ConvolutionCache> derivativeWithRespectLayerParameter(DMatrix layerInput, int variable, ConvolutionCache cache, Model.ParameterBindings bindings) {
        double[] derivative = new double[outputLength()];
        for (int convIndex = 0; convIndex < convolutions.length; convIndex++) {
            for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                ChannelCache cacheEntry = cache.channels()[convIndex][channelIndex];
                DMatrix activationWithRespectConvolution = cacheEntry.activationWithRespectConvolution();
                ConvolutionFilterMatrix matrix = convolutions[convIndex].matrix();
                DMatrix convolutionDerivative =
                        new MatrixVectorProduct(
                                new MatrixProduct(
                                        new DMatrixExpression(activationWithRespectConvolution),
                                        new DMatrixExpression(matrix.computePartialDerivative(bindings, variable))
                                ),
                                new DMatrixColumnVectorExpression(layerInput)
                        ).evaluate(bindings);
                int dstOffset = (convIndex * inputChannels + channelIndex) * convolutions[0].inputLength();
                for (int i = 0; i < convolutionDerivative.getNumRows(); i++) {
                    derivative[dstOffset + i] = convolutionDerivative.get(i, 0);
                }
            }
        }

        return new Result<>(new DMatrixRMaj(derivative.length, 1, true, derivative), cache);
    }

    @Override
    public DMatrix getEvaluation(ConvolutionCache cache) {
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

    public record ConvolutionCache(DMatrix output, ChannelCache[][] channels) {}

    public record ChannelCache(DMatrix activation, DMatrix activationInput, DMatrix activationWithRespectConvolution) {}
}
