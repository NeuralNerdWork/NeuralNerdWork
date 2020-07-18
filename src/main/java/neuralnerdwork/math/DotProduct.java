package neuralnerdwork.math;

public record DotProduct(VectorExpression left, VectorExpression right) implements ScalarExpression {
    public static ScalarExpression product(VectorExpression left, VectorExpression right) {
        final DotProduct product = new DotProduct(left, right);
        if (product.isZero()) {
            return new ConstantScalar(0.0);
        } else {
            return product;
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        final Vector lVector = left.evaluate(bindings);
        final Vector rVector = right.evaluate(bindings);

        double accum = 0.0;
        // TODO check matching length and throw with message
        for (int i = 0; i < Math.max(lVector.length(), rVector.length()); i++) {
            accum += lVector.get(i) * rVector.get(i);
        }

        return accum;
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        final MatrixExpression leftDerivative = left.computeDerivative(bindings, variables);
        final MatrixExpression rightDerivative = right.computeDerivative(bindings, variables);

        return VectorSum.sum(MatrixVectorProduct.product(new TransposeExpression(leftDerivative), right),
                             MatrixVectorProduct.product(new TransposeExpression(rightDerivative), left)).evaluate(bindings);
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final Vector leftDerivative = left.computePartialDerivative(bindings, variable);
        final Vector rightDerivative = right.computePartialDerivative(bindings, variable);

        return ScalarSum.sum(
                DotProduct.product(leftDerivative, right),
                DotProduct.product(left, rightDerivative)
        ).evaluate(bindings);
    }
}
