package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

public record TransposeExpression(MatrixExpression matrix) implements MatrixExpression {
    @Override
    public int rows() {
        return matrix.cols();
    }

    @Override
    public int cols() {
        return matrix.rows();
    }

    @Override
    public boolean isZero() {
        return matrix.isZero();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix evaluated = matrix.evaluate(bindings);

        if (evaluated instanceof DMatrixRMaj m) {
            CommonOps_DDRM.transpose(m);
            return m;
        } else if (evaluated instanceof DMatrixSparseCSC m) {
            DMatrixSparseCSC transposed = m.createLike();
            CommonOps_DSCC.transpose(m, transposed, null);

            return transposed;
        } else {
            throw new UnsupportedOperationException("Can't transpose matrix of type " + evaluated.getClass());
        }
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new TransposeExpression(new DMatrixExpression(matrix.computePartialDerivative(bindings, variable))).evaluate(bindings);
    }
}
