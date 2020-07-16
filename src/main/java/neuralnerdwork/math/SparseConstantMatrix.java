package neuralnerdwork.math;

import java.util.Iterator;
import java.util.Map;

public record SparseConstantMatrix(Map<Index, Double> values, int rows, int cols) implements Matrix {
    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
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

    public Iterable<Map.Entry<Index, Double>> entries() {
        return values.entrySet();
    }

    public static record Index(int row, int col) {}
}
