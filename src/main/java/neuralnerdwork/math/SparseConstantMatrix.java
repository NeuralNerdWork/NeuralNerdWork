package neuralnerdwork.math;

import java.util.Map;

public record SparseConstantMatrix(Map<Index, Double> values, int rows, int cols) implements MatrixExpression, Matrix {
    @Override
    public Matrix evaluate(Model.Binder bindings) {
        return this;
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    @Override
    public boolean isZero() {
        return values.values()
                     .stream()
                     .mapToDouble(n -> n)
                     .allMatch(n -> n == 0.0);
    }

    @Override
    public double get(int row, int col) {
        return values.getOrDefault(new Index(row, col), 0.0);
    }

    public static record Index(int row, int col) {}
}
