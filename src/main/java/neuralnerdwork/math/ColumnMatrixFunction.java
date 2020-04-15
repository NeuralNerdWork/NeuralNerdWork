package neuralnerdwork.math;

import java.util.Arrays;

public record ColumnMatrixFunction(VectorFunction[] columns) implements MatrixFunction {
    public ColumnMatrixFunction {
        if (columns == null || columns.length == 0) {
            throw new IllegalArgumentException("Cannot have column matrix with empty columns");
        }
        final long count = Arrays.stream(columns)
                                 .map(VectorFunction::length)
                                 .distinct()
                                 .count();
        if (count != 1) {
            throw new IllegalArgumentException("All columns must be same length");
        }
    }

    @Override
    public int inputLength() {
        return Arrays.stream(columns)
                     .mapToInt(VectorFunction::inputLength)
                     .max()
                     .orElseThrow();
    }

    @Override
    public int rows() {
        return columns[0].length();
    }

    @Override
    public int cols() {
        return columns.length;
    }

    @Override
    public Matrix apply(double[] inputs) {
        final double[][] values = new double[rows()][cols()];
        for (int col = 0; col < cols(); col++) {
            final Vector vector = columns[col].apply(inputs);
            for (int row = 0; row < rows(); row++) {
                values[row][col] = vector.get(row);
            }
        }

        return new ConstantMatrix(values);
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }
}
