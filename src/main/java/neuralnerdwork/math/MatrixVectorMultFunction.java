package neuralnerdwork.math;

public record MatrixVectorMultFunction(MatrixFunction left, VectorFunction right) implements VectorFunction {
    @Override
    public Vector apply(double[] input) {
        final Matrix leftValue = left.apply(input);
        final Vector rightValue = right.apply(input);
        assert leftValue.rows() == rightValue.length();

        final double[] values = new double[leftValue.rows()];
        for (int row = 0; row < leftValue.rows(); row++) {
            for (int col = 0; col < leftValue.cols(); col++) {
                values[row] += leftValue.get(row, col) * rightValue.get(col);
            }
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        // Uses product rule
        // (Fg)' = F'g + Fg'
        // TODO optimize for constant case

        final MatrixFunction leftDerivative = left.differentiate(variableIndex);
        final VectorFunction rightDerivative = right.differentiate(variableIndex);

        return new VectorSumFunction(
                new MatrixVectorMultFunction(
                        leftDerivative,
                        right
                ),
                new MatrixVectorMultFunction(
                        left,
                        rightDerivative
                )
        );
    }
}
