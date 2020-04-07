package neuralnerdwork.math;

public record MatrixVectorProductFunction(MatrixFunction left, VectorFunction right) implements VectorFunction {
    public MatrixVectorProductFunction {
        if (left.cols() != right.length()) {
            throw new IllegalArgumentException("Matrix width must match vector height for multiplication");
        }
    }

    @Override
    public int length() {
        return left.rows();
    }

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
                new MatrixVectorProductFunction(
                        leftDerivative,
                        right
                ),
                new MatrixVectorProductFunction(
                        left,
                        rightDerivative
                )
        );
    }
}
