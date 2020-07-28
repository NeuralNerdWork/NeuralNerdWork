package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record MatrixRowConcat(MatrixExpression top,
                              MatrixExpression bottom) implements MatrixExpression {
    public MatrixRowConcat {
        if (top.cols() != bottom.cols()) {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public int rows() {
        return top.rows() + bottom.rows();
    }

    @Override
    public int cols() {
        return top.cols();
    }

    @Override
    public boolean isZero() {
        return top.isZero() && bottom.isZero();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix leftEval = top.evaluate(bindings);
        final DMatrix rightEval = bottom.evaluate(bindings);
        final DMatrixRMaj values = new DMatrixRMaj(rows(), cols());

        for (int i = 0; i < leftEval.getNumRows(); i++) {
            for (int j = 0; j < leftEval.getNumCols(); j++) {
                values.set(i, j, leftEval.get(i, j));
            }
        }
        for (int i = 0; i < rightEval.getNumRows(); i++) {
            for (int j = 0; j < rightEval.getNumCols(); j++) {
                values.set(leftEval.getNumRows() + i, j, rightEval.get(i, j));
            }
        }

        return values;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final DMatrix top = this.top.computePartialDerivative(bindings, variable);
        final DMatrix bottom = this.bottom.computePartialDerivative(bindings, variable);

        return new MatrixRowConcat(new DMatrixExpression(top), new DMatrixExpression(bottom)).evaluate(bindings);
    }
}
