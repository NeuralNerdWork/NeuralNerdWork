package neuralnerdwork.math;

import org.ejml.data.DMatrix;

import java.util.stream.StreamSupport;

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
        final DMatrix leftValue = this.left.evaluate(bindings);
        final Vector rightValue = this.right.evaluate(bindings);

        final double[] values = new double[leftValue.getNumRows()];
        for (int col = 0; col < leftValue.getNumCols(); col++) {
            if (rightValue.get(col) != 0.0) {
                for (int row = 0; row < leftValue.getNumRows(); row++) {
                    values[row] += leftValue.get(row, col) * rightValue.get(col);
                }
            }
        }

        return new ConstantVector(values);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final VectorExpression[] leftColumns =
                StreamSupport.stream(bindings.variables().spliterator(), false)
                             .map(variable -> left.computePartialDerivative(bindings, variable))
                             .map(derivativeMatrix -> MatrixVectorProduct.product(new DMatrixExpression(derivativeMatrix), right))
                             .toArray(VectorExpression[]::new);

        final DMatrix rightDerivative = right.computeDerivative(bindings);
        final MatrixExpression rightOfSum = MatrixProduct.product(left, new DMatrixExpression(rightDerivative));
        final ColumnMatrix columnMatrix = new ColumnMatrix(leftColumns);

        return MatrixSum.sum(
                columnMatrix,
                rightOfSum
        ).evaluate(bindings);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        // Uses product rule
        // (Fg)' = F'g + Fg'

        final DMatrix leftDerivative = left.computePartialDerivative(bindings, variable);
        if (right instanceof ConstantVector) {
            return MatrixVectorProduct.product(
                    new DMatrixExpression(leftDerivative),
                    right
            ).evaluate(bindings);
        } else {
            final Vector rightDerivative = right.computePartialDerivative(bindings, variable);

            return VectorSum.sum(
                    MatrixVectorProduct.product(
                            new DMatrixExpression(leftDerivative),
                            right
                    ),
                    MatrixVectorProduct.product(
                            left,
                            rightDerivative
                    )
            ).evaluate(bindings);
        }
    }
}
