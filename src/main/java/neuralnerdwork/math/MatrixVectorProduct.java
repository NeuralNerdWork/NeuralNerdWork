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
    public Vector evaluate(Model.ParameterBindings bindings) {
        final Matrix leftValue = this.left.evaluate(bindings);
        final Vector rightValue = this.right.evaluate(bindings);

        final double[] values = new double[leftValue.rows()];
        if (leftValue instanceof SparseConstantMatrix scm) {
            for (var e : scm.entries()) {
                final SparseConstantMatrix.Index index = e.getKey();
                final double value = e.getValue();
                values[index.row()] += rightValue.get(index.col()) * value;
            }
        } else {
            for (int col = 0; col < leftValue.cols(); col++) {
                if (rightValue.get(col) != 0.0) {
                    for (int row = 0; row < leftValue.rows(); row++) {
                        values[row] += leftValue.get(row, col) * rightValue.get(col);
                    }
                }
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

        final MatrixExpression leftDerivative = left.computePartialDerivative(variable);
        if (right instanceof ConstantVector) {
            return MatrixVectorProduct.product(
                    leftDerivative,
                    right
            );
        } else {
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
}
