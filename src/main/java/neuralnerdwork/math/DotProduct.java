package neuralnerdwork.math;

public record DotProduct(VectorExpression left, VectorExpression right) implements ScalarExpression {
    public static ScalarExpression product(VectorExpression left, VectorExpression right) {
        final DotProduct product = new DotProduct(left, right);
        if (product.isZero()) {
            return new ScalarConstant(0.0);
        } else {
            return product;
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public double evaluate(Model.Binder bindings) {
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
    public VectorExpression computeDerivative(int[] variables) {
        final MatrixExpression leftDerivative = left.computeDerivative(variables);
        final MatrixExpression rightDerivative = right.computeDerivative(variables);

        return VectorSum.sum(MatrixVectorProduct.product(new Transpose(leftDerivative), right),
                             MatrixVectorProduct.product(new Transpose(rightDerivative), left));
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        final VectorExpression leftDerivative = left.computePartialDerivative(variable);
        final VectorExpression rightDerivative = right.computePartialDerivative(variable);

        return ScalarSum.sum(
                DotProduct.product(leftDerivative, right),
                DotProduct.product(left, rightDerivative)
        );
    }
}
