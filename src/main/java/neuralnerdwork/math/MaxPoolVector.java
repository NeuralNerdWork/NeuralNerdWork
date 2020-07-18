package neuralnerdwork.math;

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
    public Vector evaluate(Model.ParameterBindings bindings) {
        Vector source = input.evaluate(bindings);

        int targetWidth = this.inputWidth / this.filterWidth;
        int targetHeight = this.inputHeight / this.filterHeight;
        int length = targetWidth * targetHeight;
        double[] values = new double[length];

        for (int targetRow = 0; targetRow < targetHeight; targetRow++) {
            for (int targetCol = 0; targetCol < targetWidth; targetCol++) {
                int sourceRow = targetRow * filterHeight;
                int sourceCol = targetCol * filterWidth;
                int targetIndex = targetRow * targetWidth + targetCol;
                values[targetIndex] = source.get(sourceRow * inputWidth + sourceCol);
                for (int rowOffset = 0; rowOffset < filterHeight; rowOffset++) {
                    for (int colOffset = 0; colOffset < filterWidth; colOffset++) {
                        double candidate = source.get((sourceRow + rowOffset) * inputWidth + (sourceCol + colOffset));
                        if (candidate > values[targetIndex]) {
                            values[targetIndex] = candidate;
                        }
                    }
                }
            }
        }

        return new ConstantVector(values);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        // Need to refactor derivative to involve parameter bindings
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        // Need to refactor derivative to involve parameter bindings
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
