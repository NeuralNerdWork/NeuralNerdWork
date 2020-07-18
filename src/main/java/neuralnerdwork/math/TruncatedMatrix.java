package neuralnerdwork.math;

public record TruncatedMatrix(MatrixExpression expression, int rows, int cols) implements MatrixExpression {

    public TruncatedMatrix {
        if (expression.rows() < rows || expression.cols() < cols) {
            throw new IllegalArgumentException(String.format("Cannot truncate %dx%d matrix to %dx%d",
                                                             expression.rows(), expression.cols(), rows, cols));
        }
        if (rows == 0 || cols == 0) {
            throw new IllegalArgumentException("Cannot set rows or cols to zero");
        }
    }

    @Override
    public boolean isZero() {
        return expression.isZero();
    }

    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
        final Matrix matrix = expression.evaluate(bindings);
        return new Matrix() {
            @Override
            public double get(int row, int col) {
                if (row < matrix.rows() && col < matrix.cols()) {
                    return matrix.get(row, col);
                } else {
                    throw new IndexOutOfBoundsException(String.format("(row, col) = (%d, %d) is out of bounds for %dx%d matrix",
                                                                      row, col, matrix.rows(), matrix.cols()));
                }
            }

            @Override
            public int rows() {
                return rows;
            }

            @Override
            public int cols() {
                return cols;
            }
        };
    }

    @Override
    public Matrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new TruncatedMatrix(expression.computePartialDerivative(bindings, variable), rows, cols).evaluate(bindings);
    }
}
