package neuralnerdwork.math;

public record MatrixRowConcat(MatrixExpression top,
                              MatrixExpression bottom) implements MatrixExpression {
    public MatrixRowConcat {
        if (top.cols() != bottom.cols()) {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public int rows() {
        return top.rows() + bottom.rows();
    }

    @Override
    public int cols() {
        return top.cols();
    }

    @Override
    public boolean isZero() {
        return top.isZero() && bottom.isZero();
    }

    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
        final Matrix leftEval = top.evaluate(bindings);
        final Matrix rightEval = bottom.evaluate(bindings);
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
        final MatrixExpression leftDerivative = top.computePartialDerivative(variable);
        final MatrixExpression rightDerivative = bottom.computePartialDerivative(variable);

        return new MatrixRowConcat(leftDerivative, rightDerivative);
    }
}
