package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

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

        if (leftMatrix instanceof DMatrixRMaj l && rightMatrix instanceof DMatrixRMaj r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DDRM.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixSparseCSC l && rightMatrix instanceof DMatrixSparseCSC r) {
            DMatrixSparseCSC retVal = new DMatrixSparseCSC(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixSparseCSC l && rightMatrix instanceof DMatrixRMaj r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixRMaj l && rightMatrix instanceof DMatrixSparseCSC r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.multTransAB(r, l, retVal);
            CommonOps_DDRM.transpose(retVal);

            return retVal;
        } else {
            throw new UnsupportedOperationException("Cannot multiply matrix types " + leftMatrix.getClass() + " and " + rightMatrix.getClass());
        }
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
