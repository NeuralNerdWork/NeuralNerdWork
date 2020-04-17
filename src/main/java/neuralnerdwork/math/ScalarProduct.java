package neuralnerdwork.math;

public record ScalarProduct(ScalarExpression left,
                            ScalarExpression right) implements ScalarExpression {

    public static ScalarExpression product(ScalarExpression left,
                                           ScalarExpression right) {
        final ScalarProduct product = new ScalarProduct(left, right);
        if (product.isZero()) {
            return new ScalarConstant(0.0);
        } else {
            return product;
        }
    }

    @Override
    public double evaluate(Model.Binder bindings) {
        return left.evaluate(bindings) * right.evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        final ScalarExpression leftDerivative = left.computePartialDerivative(variable);
        final ScalarExpression rightDerivative = right.computePartialDerivative(variable);

        // Product rule
        // (fg)' = f'g + fg'
        return ScalarSum.sum(
                ScalarProduct.product(
                        leftDerivative,
                        right
                ),
                ScalarProduct.product(
                        left,
                        rightDerivative
                )
        );
    }
}