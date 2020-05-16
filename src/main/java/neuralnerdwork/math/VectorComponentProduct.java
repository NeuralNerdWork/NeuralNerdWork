package neuralnerdwork.math;

public record VectorComponentProduct(VectorExpression left,
                                     VectorExpression right) implements VectorExpression {
    public static VectorExpression product(VectorExpression left,
                                                 VectorExpression right) {
        final VectorComponentProduct product = new VectorComponentProduct(left, right);
        if (product.isZero()) {
            return new ConstantVector(new double[product.length()]);
        } else {
            return product;
        }
    }

    public VectorComponentProduct {
        if (left.length() != right.length()) {
            throw new IllegalArgumentException(String.format("left and right must have same lengths, but found %d != %d",
                                                             left.length(),
                                                             right.length()));
        }
    }

    @Override
    public int length() {
        return left.length();
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final Vector leftEval = left.evaluate(bindings);
        final Vector rightEval = right.evaluate(bindings);

        final double[] values = new double[length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = leftEval.get(i) * rightEval.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return VectorSum.sum(
                VectorComponentProduct.product(
                        left.computePartialDerivative(variable),
                        right
                ),
                VectorComponentProduct.product(
                        left,
                        right.computePartialDerivative(variable)
                )
        );
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        final MatrixExpression leftDerivative = left.computeDerivative(variables);
        final MatrixExpression rightDerivative = right.computeDerivative(variables);

        return MatrixSum.sum(
                MatrixProduct.product(
                        new DiagonalizedVector(right),
                        leftDerivative
                ),
                MatrixProduct.product(
                        new DiagonalizedVector(left),
                        rightDerivative
                )
        );
    }
}
