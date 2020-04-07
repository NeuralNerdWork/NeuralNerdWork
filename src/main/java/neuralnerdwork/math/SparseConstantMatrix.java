package neuralnerdwork.math;

import java.util.Map;

public record SparseConstantMatrix(Map<Index, Double> values, int rows, int cols) implements MatrixFunction, Matrix {
    @Override
    public Matrix apply(double[] input) {
        return this;
    }

    @Override
    public int inputLength() {
        return 0;
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    @Override
    public double get(int row, int col) {
        return values.getOrDefault(new Index(row, col), 0.0);
    }

    public static record Index(int row, int col) {}
}
