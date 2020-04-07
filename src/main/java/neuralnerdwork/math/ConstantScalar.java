package neuralnerdwork.math;

import java.util.Set;

public record ConstantScalar(double value) implements ScalarFunction {

    @Override
    public double apply(ScalarVariableBinding[] input) {
        return value;
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        return new ConstantScalar(0.0);
    }

    @Override
    public VectorFunction differentiate(ScalarVariable[] variables) {
        return new ConstantVector(new double[variables.length]);
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of();
    }
}
