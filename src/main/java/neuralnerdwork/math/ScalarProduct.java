package neuralnerdwork.math;

import org.ejml.data.DMatrix;

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
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);

        // Product rule
        // (fg)' = f'g + fg'
        return VectorSum.sum(
                new ScaledVector(
                        right,
                        new DMatrixColumnVectorExpression(leftDerivative)
                ),
                new ScaledVector(
                        left,
                        new DMatrixColumnVectorExpression(rightDerivative)
                )
        ).evaluate(bindings);
    }
}
