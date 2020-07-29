package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record MaxPoolVector(VectorExpression input,
                            int inputWidth,
                            int inputHeight,
                            int filterWidth,
                            int filterHeight) implements VectorExpression {

    public MaxPoolVector {
        if (input.length() != inputWidth * inputHeight) {
            throw new IllegalArgumentException(String.format("Input vector length [%d] does not match input dimensions %dx%d=%d",
                                                             input.length(),
                                                             inputWidth,
                                                             inputHeight,
                                                             inputWidth * inputHeight));
        }
        if (inputWidth % filterWidth != 0 || inputHeight % filterHeight != 0) {
            throw new IllegalArgumentException(String.format("filter dimensions %dx%d must divide input dimensions %dx%d",
                                                             filterWidth,
                                                             filterHeight,
                                                             inputWidth,
                                                             inputHeight));
        }
    }

    @Override
    public int length() {
        return (inputWidth / filterWidth) * (inputHeight / filterHeight);
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        DMatrix source = input.evaluate(bindings);

        int targetWidth = this.inputWidth / this.filterWidth;
        int targetHeight = this.inputHeight / this.filterHeight;
        int length = targetWidth * targetHeight;
        double[] values = new double[length];

        for (int targetRow = 0; targetRow < targetHeight; targetRow++) {
            for (int targetCol = 0; targetCol < targetWidth; targetCol++) {
                int sourceRow = targetRow * filterHeight;
                int sourceCol = targetCol * filterWidth;
                int targetIndex = targetRow * targetWidth + targetCol;
                values[targetIndex] = source.get(sourceRow * inputWidth + sourceCol, 0);
                for (int rowOffset = 0; rowOffset < filterHeight; rowOffset++) {
                    for (int colOffset = 0; colOffset < filterWidth; colOffset++) {
                        double candidate = source.get((sourceRow + rowOffset) * inputWidth + (sourceCol + colOffset), 0);
                        if (candidate > values[targetIndex]) {
                            values[targetIndex] = candidate;
                        }
                    }
                }
            }
        }

        return new DMatrixRMaj(values.length, 1, true, values);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        DMatrix source = input.evaluate(bindings);
        DMatrix sourceDerivative = input.computePartialDerivative(bindings, variable);

        int targetWidth = this.inputWidth / this.filterWidth;
        int targetHeight = this.inputHeight / this.filterHeight;
        int length = targetWidth * targetHeight;
        double[] derivatives = new double[length];

        for (int targetRow = 0; targetRow < targetHeight; targetRow++) {
            for (int targetCol = 0; targetCol < targetWidth; targetCol++) {
                int sourceRow = targetRow * filterHeight;
                int sourceCol = targetCol * filterWidth;
                int targetIndex = targetRow * targetWidth + targetCol;
                double curMax = source.get(sourceRow * inputWidth + sourceCol, 0);
                int sourceIndex = sourceRow * inputWidth + sourceCol;
                derivatives[targetIndex] = sourceDerivative.get(sourceIndex, 0);
                for (int rowOffset = 0; rowOffset < filterHeight; rowOffset++) {
                    for (int colOffset = 0; colOffset < filterWidth; colOffset++) {
                        sourceIndex = (sourceRow + rowOffset) * inputWidth + (sourceCol + colOffset);
                        double candidate = source.get(sourceIndex, 0);
                        if (candidate > curMax) {
                            curMax = candidate;
                            derivatives[targetIndex] = sourceDerivative.get(sourceIndex, 0);
                        }
                    }
                }
            }
        }

        return new DMatrixRMaj(derivatives.length, 1, true, derivatives);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        DMatrix source = input.evaluate(bindings);
        DMatrix sourceDerivative = input.computeDerivative(bindings);

        int targetWidth = this.inputWidth / this.filterWidth;
        int targetHeight = this.inputHeight / this.filterHeight;
        int length = targetWidth * targetHeight;
        DMatrixRMaj derivative = new DMatrixRMaj(length, bindings.size());

        for (int targetRow = 0; targetRow < targetHeight; targetRow++) {
            for (int targetCol = 0; targetCol < targetWidth; targetCol++) {
                int sourceRow = targetRow * filterHeight;
                int sourceCol = targetCol * filterWidth;
                int targetIndex = targetRow * targetWidth + targetCol;
                double curMax = source.get(sourceRow * inputWidth + sourceCol, 0);
                int sourceIndex = sourceRow * inputWidth + sourceCol;
                for (int varIndex = 0; varIndex < bindings.size(); varIndex++) {
                    derivative.set(targetIndex, varIndex, sourceDerivative.get(sourceIndex, varIndex));
                }
                for (int rowOffset = 0; rowOffset < filterHeight; rowOffset++) {
                    for (int colOffset = 0; colOffset < filterWidth; colOffset++) {
                        sourceIndex = (sourceRow + rowOffset) * inputWidth + (sourceCol + colOffset);
                        double candidate = source.get(sourceIndex, 0);
                        if (candidate > curMax) {
                            curMax = candidate;
                            for (int varIndex = 0; varIndex < bindings.size(); varIndex++) {
                                derivative.set(targetIndex, varIndex, sourceDerivative.get(sourceIndex, varIndex));
                            }
                        }
                    }
                }
            }
        }

        return derivative;
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
