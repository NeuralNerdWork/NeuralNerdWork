package neuralnerdwork.math;

public record ScalarConstant(double value) implements ScalarExpression {
    @Override
    public double evaluate(Model.Binder bindings) {
        return value;
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        return new ScalarConstant(0.0);
    }

    @Override
    public boolean isZero() {
        return value == 0.0;
    }
}
