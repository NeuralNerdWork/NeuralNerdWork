package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

public record DMatrixColumnVectorExpression(DMatrix matrix) implements VectorExpression {
    public DMatrixColumnVectorExpression {
        if (matrix.getNumCols() != 1) {
            throw new IllegalArgumentException(String.format("%dx%d matrix is not a column vector", matrix.getNumRows(), matrix.getNumCols()));
        }
    }

    @Override
    public int length() {
        return matrix.getNumRows();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        return matrix;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new DMatrixSparseCSC(length(), 1, 0);
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new DMatrixSparseCSC(length(), bindings.size(), 0);
    }
}
