package neuralnerdwork.math;

import org.ejml.data.DMatrix;

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
    public Vector evaluate(Model.ParameterBindings bindings) {
        final Vector leftEval = left.evaluate(bindings);
        final Vector rightEval = right.evaluate(bindings);

        final double[] values = new double[length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = leftEval.get(i) * rightEval.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                VectorComponentProduct.product(
                        left.computePartialDerivative(bindings, variable),
                        right
                ),
                VectorComponentProduct.product(
                        left,
                        right.computePartialDerivative(bindings, variable)
                )
        ).evaluate(bindings);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);

        return MatrixSum.sum(
                MatrixProduct.product(
                        new DiagonalizedVector(right),
                        new DMatrixExpression(leftDerivative)
                ),
                MatrixProduct.product(
                        new DiagonalizedVector(left),
                        new DMatrixExpression(rightDerivative)
                )
        ).evaluate(bindings);
    }
}
