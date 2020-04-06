package neuralnerdwork.math;

import java.util.Set;

public record ConstantVector(double[] values) implements VectorFunction, Vector {

    @Override
    public double get(int index) {
        return values[index];
    }

    @Override
    public int length() {
        return values.length;
    }

    @Override
    public Vector apply(VectorVariableBinding input) {
        return this;
    }

    @Override
    public VectorFunction differentiate(ScalarVariable variable) {
        return new ConstantVector(new double[values.length]);
    }

    @Override
    public MatrixFunction differentiate(VectorVariable variable) {
        return new ConstantMatrix(new double[values.length][variable.variables().length]);
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of();
    }
}
