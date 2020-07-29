package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

public record MatrixProduct(MatrixExpression left, MatrixExpression right) implements MatrixExpression {
    public static MatrixExpression product(MatrixExpression left, MatrixExpression right) {
        final MatrixProduct product = new MatrixProduct(left, right);
        if (product.isZero()) {
            return new DMatrixExpression(new DMatrixSparseCSC(product.rows(), product.cols(), 0));
        } else {
            return product;
        }
    }

    public MatrixProduct {
        if (left.cols() != right.rows()) {
            throw new IllegalArgumentException(
                    String.format("Cannot multiply matrices of dimensions (%dx%d) and (%dx%d)",
                                  left.rows(),
                                  left.cols(),
                                  right.rows(),
                                  right.cols())
            );
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public int rows() {
        return left.rows();
    }

    @Override
    public int cols() {
        return right.cols();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix leftMatrix = left.evaluate(bindings);
        final DMatrix rightMatrix = right.evaluate(bindings);

        return EJMLUtil.mult(leftMatrix, rightMatrix);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        // This uses product rule
        // (FG)' = F'G + FG'
        final DMatrix leftDerivative = left.computePartialDerivative(bindings, variable);
        final DMatrix rightDerivative = right.computePartialDerivative(bindings, variable);

        return MatrixSum.sum(
                MatrixProduct.product(new DMatrixExpression(leftDerivative), right),
                MatrixProduct.product(left, new DMatrixExpression(rightDerivative))
        ).evaluate(bindings);
    }
}
