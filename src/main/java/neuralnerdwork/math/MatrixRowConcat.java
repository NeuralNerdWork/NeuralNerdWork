package neuralnerdwork.math;

public record MatrixRowConcat(MatrixExpression left,
                              MatrixExpression right) implements MatrixExpression {
    public MatrixRowConcat {
        if (left.cols() != right.cols()) {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public int rows() {
        return left.rows() + right.rows();
    }

    @Override
    public int cols() {
        return left.cols();
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix leftEval = left.evaluate(bindings);
        final Matrix rightEval = right.evaluate(bindings);
        final double[][] values = new double[rows()][cols()];

        for (int i = 0; i < leftEval.rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                values[i][j] = leftEval.get(i, j);
            }
        }
        for (int i = 0; i < rightEval.rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                values[leftEval.rows()+i][j] = rightEval.get(i, j);
            }
        }

        return new ConstantArrayMatrix(values);
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        final MatrixExpression leftDerivative = left.computePartialDerivative(variable);
        final MatrixExpression rightDerivative = right.computePartialDerivative(variable);

        return new MatrixRowConcat(leftDerivative, rightDerivative);
    }
}
