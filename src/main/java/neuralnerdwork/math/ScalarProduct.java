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
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final double leftDerivative = left.computePartialDerivative(bindings, variable);
        final double rightDerivative = right.computePartialDerivative(bindings, variable);

        // Product rule
        // (fg)' = f'g + fg'
        return ScalarSum.sum(
                ScalarProduct.product(
                        new ConstantScalar(leftDerivative),
                        right
                ),
                ScalarProduct.product(
                        left,
                        new ConstantScalar(rightDerivative)
                )
        ).evaluate(bindings);
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        final VectorExpression leftDerivative = left.computeDerivative(bindings, variables);
        final VectorExpression rightDerivative = right.computeDerivative(bindings, variables);

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
        ).evaluate(bindings);
    }
}
