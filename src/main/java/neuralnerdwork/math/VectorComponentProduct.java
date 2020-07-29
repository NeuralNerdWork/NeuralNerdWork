package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixD1;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.sparse.csc.CommonOps_DSCC;

public record VectorComponentProduct(VectorExpression left,
                                     VectorExpression right) implements VectorExpression {
    public static VectorExpression product(VectorExpression left, VectorExpression right) {
        final VectorComponentProduct product = new VectorComponentProduct(left, right);
        if (product.isZero()) {
            return new DMatrixColumnVectorExpression(new DMatrixSparseCSC(product.length(), 1, 0));
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
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix leftEval = left.evaluate(bindings);
        final DMatrix rightEval = right.evaluate(bindings);

        return evaluate(leftEval, rightEval);
    }

    private DMatrix evaluate(DMatrix leftEval, DMatrix rightEval) {
        if (leftEval instanceof DMatrixRMaj l && rightEval instanceof DMatrixRMaj r) {
            DMatrixD1 retVal = l.createLike();
            CommonOps_DDRM.elementMult(l, r, retVal);

            return retVal;
        } else if (leftEval instanceof DMatrixSparseCSC l && rightEval instanceof DMatrixSparseCSC r) {
            DMatrixSparseCSC retVal = new DMatrixSparseCSC(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.elementMult(l, r, retVal, null, null);

            return retVal;
        } else if (leftEval instanceof DMatrixRMaj l && rightEval instanceof DMatrixSparseCSC r) {
            return evaluate(l, ConvertDMatrixStruct.convert(r, new DMatrixRMaj(r.getNumRows(), r.getNumCols())));
        } else if (leftEval instanceof DMatrixSparseCSC l && rightEval instanceof DMatrixRMaj r) {
            return evaluate(ConvertDMatrixStruct.convert(l, new DMatrixRMaj(l.getNumRows(), l.getNumCols())), r);
        } else {
            throw new UnsupportedOperationException(String.format("Can't multiply elements of matrix types %s and %s", leftEval
                    .getClass(), rightEval.getClass()));
        }
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                VectorComponentProduct.product(
                        new DMatrixColumnVectorExpression(left.computePartialDerivative(bindings, variable)),
                        right
                ),
                VectorComponentProduct.product(
                        left,
                        new DMatrixColumnVectorExpression(right.computePartialDerivative(bindings, variable))
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
