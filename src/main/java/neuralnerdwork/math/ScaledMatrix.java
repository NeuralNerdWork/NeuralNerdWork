package neuralnerdwork.math;

public record ScaledMatrix(ScalarExpression scalarExpression, MatrixExpression matrixExpression) implements MatrixExpression {
    public ScaledMatrix(double value, MatrixExpression matrixExpression) {
        this(new ConstantScalar(value), matrixExpression);
    }

    @Override
    public int rows() {
        return matrixExpression.rows();
    }

    @Override
    public int cols() {
        return matrixExpression.cols();
    }

    @Override
    public boolean isZero() {
        return scalarExpression.isZero() || matrixExpression.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix matrix = matrixExpression.evaluate(bindings);
        final double value = scalarExpression.evaluate(bindings);
        final double[][] values = new double[matrix.rows()][matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row][col] = value * matrix.get(row, col);
            }
        }

        return new ConstantArrayMatrix(values);
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return MatrixSum.sum(
                new ScaledMatrix(scalarExpression.computePartialDerivative(variable), matrixExpression),
                new ScaledMatrix(scalarExpression, matrixExpression.computePartialDerivative(variable))
        );
    }
}
