package neuralnerdwork.math;

public record ScaledMatrix(double value, MatrixExpression matrixExpression) implements MatrixExpression {
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
        return value == 0.0 || matrixExpression.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix matrix = matrixExpression.evaluate(bindings);
        final double[][] values = new double[matrix.rows()][matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row][col] = value * matrix.get(row, col);
            }
        }

        return new ConstantMatrix(values);
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new ScaledMatrix(value, matrixExpression.computePartialDerivative(variable));
    }
}
