package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Map;

public record ConstantArrayMatrix(double[][] values, int cols) implements MatrixExpression, Matrix {
    public ConstantArrayMatrix(double[][] values) {
        this(values, values[0].length);
    }

    @Override
    public double get(int row, int col) {
        return values[row][col];
    }

    @Override
    public int rows() {
        return values.length;
    }

    @Override
    public int cols() {
        return cols;
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(values)
                     .flatMapToDouble(Arrays::stream)
                     .allMatch(n -> n == 0.0);
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        return this;
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new SparseConstantMatrix(Map.of(), rows(), cols());
    }

    @Override
    public String toString() {
        return "ConstantMatrix{" +
                "values=" + Arrays.deepToString(values) +
                '}';
    }
}
