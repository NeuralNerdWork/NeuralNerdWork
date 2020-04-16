package neuralnerdwork.math;

public record ScalarParameter(int variable) implements ScalarExpression {
    @Override
    public double evaluate(Model.Binder bindings) {
        return bindings.get(variable);
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        return new ConstantScalar(1.0);
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
