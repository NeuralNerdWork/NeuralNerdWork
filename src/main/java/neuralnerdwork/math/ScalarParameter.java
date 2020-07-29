package neuralnerdwork.math;

import org.ejml.data.DMatrix;

import java.util.stream.StreamSupport;

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
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new ScalarComponentsVector(
                StreamSupport.stream(bindings.variables().spliterator(), false)
                             .map(variable -> new ConstantScalar(computePartialDerivative(bindings, variable)))
                             .toArray(ScalarExpression[]::new)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
