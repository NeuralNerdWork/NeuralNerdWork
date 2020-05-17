package neuralnerdwork.math;

import java.util.Arrays;

import static java.lang.String.format;

public record MatrixVectorProduct(MatrixExpression left, VectorExpression right) implements VectorExpression {
    public static VectorExpression product(MatrixExpression left, VectorExpression right) {
        final MatrixVectorProduct product = new MatrixVectorProduct(left, right);
        if (product.isZero()) {
            return new ConstantVector(new double[product.length()]);
        } else {
            return product;
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    public MatrixVectorProduct {
        if (left.cols() != right.length()) {
            throw new IllegalArgumentException(format("Cannot multiply (%dx%d) matrix by (%dx1) vector",
                                                      left.rows(), left.cols(), right.length()));
        }
    }

    @Override
    public int length() {
        return left.rows();
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final Matrix leftValue = this.left.evaluate(bindings);
        final Vector rightValue = this.right.evaluate(bindings);

        final double[] values = new double[leftValue.rows()];
        for (int row = 0; row < leftValue.rows(); row++) {
            for (int col = 0; col < leftValue.cols(); col++) {
                values[row] += leftValue.get(row, col) * rightValue.get(col);
            }
        }

        return new ConstantVector(values);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        final VectorExpression[] leftColumns =
                Arrays.stream(variables)
                      .mapToObj(left::computePartialDerivative)
                      .map(derivativeMatrix -> MatrixVectorProduct.product(derivativeMatrix, right))
                      .toArray(VectorExpression[]::new);

        final MatrixExpression rightDerivative = right.computeDerivative(variables);
        final MatrixExpression rightOfSum = MatrixProduct.product(left, rightDerivative);
        final ColumnMatrix columnMatrix = new ColumnMatrix(leftColumns);

        return MatrixSum.sum(
                columnMatrix,
                rightOfSum
        );
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        // Uses product rule
        // (Fg)' = F'g + Fg'
        // TODO optimize for constant case

        final MatrixExpression leftDerivative = left.computePartialDerivative(variable);
        final VectorExpression rightDerivative = right.computePartialDerivative(variable);

        return VectorSum.sum(
                MatrixVectorProduct.product(
                        leftDerivative,
                        right
                ),
                MatrixVectorProduct.product(
                        left,
                        rightDerivative
                )
        );
    }
}
