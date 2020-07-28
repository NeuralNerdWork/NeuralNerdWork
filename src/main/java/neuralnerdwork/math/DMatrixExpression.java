package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public record DMatrixExpression(DMatrix matrix) implements MatrixExpression {
    @Override
    public int rows() {
        return matrix.getNumRows();
    }

    @Override
    public int cols() {
        return matrix.getNumCols();
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        return matrix;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        DMatrix retVal = matrix.createLike();
        retVal.zero();

        return retVal;
    }
}
