package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

public record DMatrixRowVectorExpression(DMatrix matrix) implements VectorExpression {
    public DMatrixRowVectorExpression {
        if (matrix.getNumRows() != 1) {
            throw new IllegalArgumentException(String.format("%dx%d matrix is not a row vector", matrix.getNumRows(), matrix.getNumCols()));
        }
    }

    @Override
    public int length() {
        return matrix.getNumCols();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        return matrix;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new DMatrixSparseCSC(1, length(), 0);
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public boolean columnVector() {
        return false;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        throw new UnsupportedOperationException("Can't compute full derivative of row vector");
    }
}
