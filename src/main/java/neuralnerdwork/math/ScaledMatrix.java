package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

public record ScaledMatrix(ScalarExpression scalarExpression, MatrixExpression matrixExpression) implements MatrixExpression {
    public ScaledMatrix(double value, MatrixExpression matrixExpression) {
        this(new ConstantScalar(value), matrixExpression);
    }

    @Override
    public int rows() {
        return matrixExpression.rows();
    }

    @Override
    public int cols() {
        return matrixExpression.cols();
    }

    @Override
    public boolean isZero() {
        return scalarExpression.isZero() || matrixExpression.isZero();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix matrix = matrixExpression.evaluate(bindings);
        final double value = scalarExpression.evaluate(bindings);

        if (matrix instanceof DMatrixRMaj m) {
            DMatrixRMaj retVal = m.createLike();
            CommonOps_DDRM.scale(value, m, retVal);

            return retVal;
        } else if (matrix instanceof DMatrixSparseCSC m) {
            DMatrixSparseCSC retVal = m.createLike();
            CommonOps_DSCC.scale(value, m, retVal);

            return retVal;
        } else {
            throw new UnsupportedOperationException("Can't scale matrix type " + matrix.getClass());
        }
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return MatrixSum.sum(
                new ScaledMatrix(scalarExpression.computePartialDerivative(bindings, variable), matrixExpression),
                new ScaledMatrix(scalarExpression, new DMatrixExpression(matrixExpression.computePartialDerivative(bindings, variable)))
        ).evaluate(bindings);
    }
}
