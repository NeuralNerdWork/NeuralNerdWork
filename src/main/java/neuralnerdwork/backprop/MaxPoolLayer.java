package neuralnerdwork.backprop;

import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DMatrixSparseTriplet;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.ConvertDMatrixStruct;

import java.util.Arrays;
import java.util.stream.IntStream;

public record MaxPoolLayer(Channel[] channels) implements Layer<MaxPoolLayer.MaxPoolCache> {
    public record Channel(int inputWidth,
                          int inputHeight,
                          int filterWidth,
                          int filterHeight) {
        public int outputLength() {
            return (inputWidth / filterWidth) * (inputHeight / filterHeight);
        }

        public int inputLength() {
            return inputHeight * inputWidth;
        }
    }

    @Override
    public ActivationFunction activation() {
        return new IdentityFunction();
    }

    @Override
    public boolean containsVariable(int variable) {
        return false;
    }

    @Override
    public int outputLength() {
        return Arrays.stream(channels)
                     .mapToInt(Channel::outputLength)
                     .sum();
    }

    @Override
    public int inputLength() {
        return Arrays.stream(channels)
                     .mapToInt(Channel::inputLength)
                     .sum();
    }

    @Override
    public IntStream variables() {
        return IntStream.empty();
    }

    @Override
    public Result<DMatrix, MaxPoolCache> derivativeWithRespectToLayerInput(DMatrix layerInput, MaxPoolCache cache, Model.ParameterBindings bindings) {
        int totalInputLength = inputLength();
        int totalOutputLength = outputLength();
        DMatrixSparseTriplet derivative = new DMatrixSparseTriplet(totalOutputLength, totalInputLength, 0);
        for (int i = 0, sourceChannelOffset = 0, targetChannelOffset = 0;
             i < channels.length;
             sourceChannelOffset += channels[i].inputLength(), targetChannelOffset += channels[i].outputLength(), i += 1) {
            Channel channel = channels[i];
            int targetWidth = channel.inputWidth() / channel.filterWidth();
            int targetHeight = channel.inputHeight() / channel.filterHeight();

            for (int targetRow = 0; targetRow < targetHeight; targetRow++) {
                for (int targetCol = 0; targetCol < targetWidth; targetCol++) {
                    int sourceRowOffset = targetRow * channel.filterHeight();
                    int sourceColOffset = targetCol * channel.filterWidth();
                    int targetIndex = targetRow * targetWidth + targetCol;
                    int sourceIndexOffset = sourceChannelOffset + sourceRowOffset * channel.inputWidth() + sourceColOffset;

                    record Max(double value, int row, int col) {}
                    Max max = new Max(layerInput.get(sourceIndexOffset, 0), targetIndex, sourceIndexOffset);
                    for (int r = 0; r < channel.filterHeight(); r++) {
                        for (int c = 0; c < channel.filterWidth(); c++) {
                            int sourceIndex = sourceIndexOffset + r * channel.inputWidth() + c;
                            double candidate = layerInput.get(sourceIndex, 0);
                            if (candidate > max.value) {
                                max = new Max(candidate, targetIndex, sourceIndexOffset);
                            }
                        }
                    }
                    derivative.addItem(max.row, max.col, max.value);
                }
            }
        }

        return new Result<>(ConvertDMatrixStruct.convert(derivative, (DMatrixSparseCSC) null), cache);
    }

    @Override
    public Result<DMatrix, MaxPoolCache> derivativeWithRespectLayerParameter(DMatrix layerInput, int variable, MaxPoolCache cache, Model.ParameterBindings bindings) {
        throw new UnsupportedOperationException("Max pool layers don't contain any variables. This should never be called.");
    }

    @Override
    public Result<DMatrix, MaxPoolCache> evaluate(DMatrix layerInput, Model.ParameterBindings bindings) {
        DMatrixRMaj accum = new DMatrixRMaj(outputLength(), 1);
        for (int i = 0, inputChannelOffset = 0, outputChannelOffset = 0;
             i < channels.length;
             inputChannelOffset += channels[i].inputLength(), outputChannelOffset += channels[i].outputLength(), i += 1) {
            Channel channel = channels[i];
            DMatrixRMaj channelInput = new DMatrixRMaj(channel.inputLength(), 1);
            CommonOps_DDRM.extract(layerInput, inputChannelOffset, 0, channelInput);
            MaxPoolVector maxPoolVector = new MaxPoolVector(new DMatrixColumnVectorExpression(channelInput),
                                                            channel.inputWidth,
                                                            channel.inputHeight,
                                                            channel.filterWidth,
                                                            channel.filterHeight);
            DMatrix channelOutput = maxPoolVector.evaluate(bindings);
            CommonOps_DDRM.insert(channelOutput, accum, outputChannelOffset, 0);
        }

        return new Result<>(accum, new MaxPoolCache(accum));
    }

    @Override
    public DMatrix getEvaluation(MaxPoolCache cache) {
        return cache.evaluated();
    }

    public record MaxPoolCache(DMatrix evaluated) {}
}
