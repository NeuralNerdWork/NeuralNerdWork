package neuralnerdwork.math;

public record ScalarConstantMultiple(double constant, ScalarExpression expression) implements ScalarExpression {
    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return constant * expression.evaluate(bindings);
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        return ScalarProduct.product(new ConstantScalar(constant), expression.computePartialDerivative(variable));
    }

    @Override
    public VectorExpression computeDerivative(int[] variables) {
        return new ScaledVector(
                constant,
                expression.computeDerivative(variables)
        );
    }

    @Override
    public boolean isZero() {
        return constant == 0.0;
    }
}
