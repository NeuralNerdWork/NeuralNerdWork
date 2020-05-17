package neuralnerdwork.math;

public record ZeroPaddedMatrix(MatrixExpression expression, int extraRows, int extraColumns) implements MatrixExpression {
    @Override
    public int rows() {
        return expression.rows() + extraRows;
    }

    @Override
    public int cols() {
        return expression.cols() + extraColumns;
    }

    @Override
    public boolean isZero() {
        return expression.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix matrix = expression.evaluate(bindings);
        return new Matrix() {
            @Override
            public double get(int row, int col) {
                if (row < matrix.rows() && col < matrix.cols()) {
                    return matrix.get(row, col);
                } else if (row < rows() && col < cols()) {
                    return 0.0;
                } else {
                    throw new IllegalArgumentException(String.format("Invalid in (%d, %d) in %dx%d matrix", row, col, rows(), cols()));
                }
            }

            @Override
            public int rows() {
                return ZeroPaddedMatrix.this.rows();
            }

            @Override
            public int cols() {
                return ZeroPaddedMatrix.this.cols();
            }
        };
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        throw new RuntimeException("Not yet implemented!");
    }
}
