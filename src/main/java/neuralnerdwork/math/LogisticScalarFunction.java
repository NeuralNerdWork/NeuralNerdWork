package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Set;

public record LogisticScalarFunction(ScalarVariable variable) implements ScalarFunction {

    @Override
    public double apply(ScalarVariableBinding[] input) {
        final ScalarVariableBinding scalarVariableBinding = Arrays.stream(input)
                                                                  .filter(binding -> binding.variable().equals(variable))
                                                                  .findFirst()
                                                                  .orElseThrow(() -> new IllegalArgumentException(String.format("Cannot apply function without binding for variable %s", variable.symbol())));

        final double value = scalarVariableBinding.value();
        return logistic(value);
    }

    private double logistic(double value) {
        final double eToX = Math.exp(value);
        return eToX / (1 + eToX);
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        if (!this.variable.equals(variable)) {
            return new ConstantScalar(0.0);
        } else {
            return new ScalarMultiplicationFunction(this, new ScalarComposition(new NegateScalar(this.variable), this));
        }
    }

    @Override
    public VectorFunction differentiate(ScalarVariable[] variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of(variable);
    }
}
