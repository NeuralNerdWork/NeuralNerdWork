package neuralnerdwork.math;

public record ScalarConstantMultiple(double constant, ScalarExpression expression) implements ScalarExpression {
    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return constant * expression.evaluate(bindings);
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return ScalarProduct
                .product(new ConstantScalar(constant), new ConstantScalar(expression.computePartialDerivative(bindings, variable)))
                .evaluate(bindings);
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings) {
        return new ScaledVector(
                constant,
                expression.computeDerivative(bindings)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return constant == 0.0;
    }
}
