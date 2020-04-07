package neuralnerdwork.math;

import java.util.Map;

public record ParameterMatrix(int variableStartIndex, int rows, int cols) implements MatrixFunction {
    @Override
    public Matrix apply(double[] input) {
        final int length = rows * cols;
        final double[][] values = new double[rows][cols];
        for (int i = 0; i < length; i++) {
            final int row = i / cols;
            final int col = i % cols;

            values[row][col] = input[variableStartIndex + i];
        }

        return new ConstantMatrix(values);
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        final int length = rows * cols;
        if (variableIndex >= variableStartIndex && variableIndex < variableStartIndex + length) {
            final int row = (variableIndex - variableStartIndex) / cols;
            final int col = (variableIndex - variableStartIndex) % cols;
            return new SparseConstantMatrix(Map.of(new SparseConstantMatrix.Index(row, col), 1.0), rows, cols);
        }
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    public int indexFor(int row, int col) {
        return variableStartIndex + row * cols + col;
    }
}
