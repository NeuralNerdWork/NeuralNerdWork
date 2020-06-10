package neuralnerdwork.math;

import java.util.Map;

public record ParameterMatrix(int variableStartIndex, int rows, int cols) implements MatrixExpression {
    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
        final int length = this.rows * this.cols;
        final double[][] values = new double[this.rows][this.cols];
        for (int i = 0; i < length; i++) {
            final int row = i / this.cols;
            final int col = i % this.cols;

            values[row][col] = bindings.get(this.variableStartIndex + i);
        }

        return new ConstantArrayMatrix(values);
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        final int length = rows * cols;
        if (variable >= variableStartIndex && variable < variableStartIndex + length) {
            final int row = (variable - variableStartIndex) / cols;
            final int col = (variable - variableStartIndex) % cols;
            return new SparseConstantMatrix(Map.of(new SparseConstantMatrix.Index(row, col), 1.0), rows, cols);
        }
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    public int variableIndexFor(int row, int col) {
        return variableStartIndex + row * cols + col;
    }

    public boolean containsVariable(int variable) {
        return variable >= variableStartIndex && variable < variableStartIndex + rows * cols;
    }

    public int rowIndexFor(int variable) {
        return (variable - variableStartIndex) / cols;
    }

    public int colIndexFor(int variable) {
        return (variable - variableStartIndex) % cols;
    }
}
