package neuralnerdwork.math;

import org.ejml.data.*;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.sparse.csc.CommonOps_DSCC;

import java.util.Arrays;
import java.util.Iterator;

public record ScaledVector(ScalarExpression scalarExpression, VectorExpression vectorExpression) implements VectorExpression {
    public ScaledVector(double scalar, VectorExpression vectorExpression) {
        this(new ConstantScalar(scalar), vectorExpression);
    }

    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix vector = vectorExpression.evaluate(bindings);
        final double value = scalarExpression.evaluate(bindings);

        if (vector instanceof DMatrixRMaj m) {
            DMatrixRMaj retVal = m.createLike();
            CommonOps_DDRM.scale(value, m, retVal);

            return retVal;
        } else if (vector instanceof DMatrixSparseCSC m) {
            DMatrixSparseCSC retVal = m.createLike();
            CommonOps_DSCC.scale(value, m, retVal);

            return retVal;
        } else {
            throw new UnsupportedOperationException("Cannot scale matrix of type " + vector.getClass());
        }
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        DMatrix scalarDerivative = this.scalarExpression.computeDerivative(bindings);
        DMatrix vector = vectorExpression.evaluate(bindings);
        DMatrix scaledColumns;
        if (vector instanceof DMatrixRMaj v && scalarDerivative instanceof DMatrixRMaj sd) {
            DMatrix1Row retVal = new DMatrixRMaj(v.getNumRows(), sd.getNumCols());
            CommonOps_DDRM.mult(v, sd, retVal);

            scaledColumns = retVal;
        } else if (vector instanceof DMatrixRMaj v && scalarDerivative instanceof DMatrixSparseCSC sd) {
            DMatrix1Row retVal = new DMatrixRMaj(v.getNumRows(), sd.getNumCols());
            CommonOps_DDRM.mult(v, ConvertDMatrixStruct.convert(sd, (DMatrixRMaj) null), retVal);

            scaledColumns = retVal;
        } else {
            throw new UnsupportedOperationException(String.format("Cannot compute derivative of scaled vector for vector type %s and scalarDerivative type %s", vector
                    .getClass(), scalarDerivative.getClass()));
        }

        return MatrixSum.sum(
            new ScaledMatrix(scalarExpression, new DMatrixExpression(vectorExpression.computeDerivative(bindings))),
            new DMatrixExpression(scaledColumns)
        ).evaluate(bindings);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                new ScaledVector(scalarExpression, new DMatrixColumnVectorExpression(vectorExpression.computePartialDerivative(bindings, variable))),
                new ScaledVector(scalarExpression.computePartialDerivative(bindings, variable), vectorExpression)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return scalarExpression.isZero() || vectorExpression.isZero();
    }
}
