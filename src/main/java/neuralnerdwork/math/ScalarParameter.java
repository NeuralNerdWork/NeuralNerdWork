package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarParameter(int variable) implements ScalarExpression {
    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return bindings.get(variable);
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        if (this.variable == variable) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        return new ScalarComponentsVector(
                Arrays.stream(variables)
                      .mapToObj(variable -> new ConstantScalar(computePartialDerivative(bindings, variable)))
                      .toArray(ScalarExpression[]::new)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
