package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarParameter(int variable) implements ScalarExpression {
    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return bindings.get(variable);
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        if (this.variable == variable) {
            return new ConstantScalar(1.0);
        } else {
            return new ConstantScalar(0.0);
        }
    }

    @Override
    public VectorExpression computeDerivative(int[] variables) {
        return new ScalarComponentsVector(
                Arrays.stream(variables)
                      .mapToObj(this::computePartialDerivative)
                      .toArray(ScalarExpression[]::new)
        );
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
