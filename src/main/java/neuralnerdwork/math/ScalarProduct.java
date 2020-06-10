package neuralnerdwork.math;

public record ScalarProduct(ScalarExpression left,
                            ScalarExpression right) implements ScalarExpression {

    public static ScalarExpression product(ScalarExpression left,
                                           ScalarExpression right) {
        final ScalarProduct product = new ScalarProduct(left, right);
        if (product.isZero()) {
            return new ConstantScalar(0.0);
        } else {
            return product;
        }
    }

    @Override
    public double evaluate(Model.ParameterBindings bindings) {
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

    @Override
    public VectorExpression computeDerivative(int[] variables) {
        final VectorExpression leftDerivative = left.computeDerivative(variables);
        final VectorExpression rightDerivative = right.computeDerivative(variables);

        // Product rule
        // (fg)' = f'g + fg'
        return VectorSum.sum(
                new ScaledVector(
                        right,
                        leftDerivative
                ),
                new ScaledVector(
                        left,
                        rightDerivative
                )
        );
    }
}
