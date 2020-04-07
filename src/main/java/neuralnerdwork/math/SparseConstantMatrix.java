package neuralnerdwork.math;

import java.util.Map;
import java.util.Set;

public record SparseConstantMatrix(Map<Index, Double> values, int rows, int cols) implements MatrixFunction, Matrix {
    @Override
    public Matrix apply(ScalarVariableBinding[] input) {
        return this;
    }

    @Override
    public MatrixFunction differentiate(ScalarVariable argument) {
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of();
    }

    @Override
    public double get(int row, int col) {
        return values.getOrDefault(new Index(row, col), 0.0);
    }

    public static record Index(int row, int col) {}
}
