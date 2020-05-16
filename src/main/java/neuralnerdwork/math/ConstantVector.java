package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Map;

public record ConstantVector(double[] values) implements VectorExpression, Vector {

    @Override
    public double get(int index) {
        return values[index];
    }

    @Override
    public int length() {
        return values.length;
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(values)
                     .allMatch(n -> n == 0.0);
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        return this;
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return new ConstantVector(new double[values.length]);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return new SparseConstantMatrix(Map.of(), values.length, variables.length);
    }

    @Override
    public String toString() {
        return "ConstantVector{" +
                "values=" + Arrays.toString(values) +
                '}';
    }
}
