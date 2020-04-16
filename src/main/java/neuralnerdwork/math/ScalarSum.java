package neuralnerdwork.math;

public record ScalarSum(ScalarExpression left, ScalarExpression right) implements ScalarExpression {
    public static ScalarExpression sum(ScalarExpression left, ScalarExpression right) {
        final ScalarSum sum = new ScalarSum(left, right);
        if (sum.isZero()) {
            return new ScalarConstant(0.0);
        } else if (left.isZero()) {
            return right;
        } else if (right.isZero()) {
            return left;
        } else {
            return sum;
        }
    }

    @Override
    public double evaluate(Model.Binder bindings) {
        return left.evaluate(bindings) + right.evaluate(bindings);
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        return ScalarSum.sum(left.computePartialDerivative(variable), right.computePartialDerivative(variable));
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }
}
