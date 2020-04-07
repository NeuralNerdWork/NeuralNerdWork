package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Set;

public record NegateScalar(ScalarVariable variable) implements ScalarFunction {
    @Override
    public double apply(VectorVariableBinding input) {
        final int indexOfArgument = Arrays.asList(input.variable().variables())
                                          .indexOf(variable);
        if (indexOfArgument < 0) {
            throw new IllegalArgumentException(String.format("Cannot invoke function without variable %s", variable.symbol()));
        }

        final double value = input.value().get(indexOfArgument);
        return -value;
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        if (this.variable.equals(variable)) {
            return new ConstantScalar(-1.0);
        } else {
            return new ConstantScalar(0.0);
        }
    }

    @Override
    public VectorFunction differentiate(VectorVariable variable) {
        // Needs to return -1 for arguments that are this functions variable and 0 for others
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of(variable);
    }
}
