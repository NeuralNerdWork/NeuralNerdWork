package neuralnerdwork.math;

public record TransposeExpression(MatrixExpression matrix) implements MatrixExpression {
    @Override
    public int rows() {
        return matrix.cols();
    }

    @Override
    public int cols() {
        return matrix.rows();
    }

    @Override
    public boolean isZero() {
        return matrix.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix evaluated = matrix.evaluate(bindings);

        return new Matrix() {
            @Override
            public double get(int row, int col) {
                return evaluated.get(col, row);
            }

            @Override
            public int rows() {
                return evaluated.cols();
            }

            @Override
            public int cols() {
                return evaluated.rows();
            }
        };
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new TransposeExpression(matrix.computePartialDerivative(variable));
    }
}
