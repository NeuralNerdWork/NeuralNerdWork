package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record RowVectorConcat(VectorExpression left, VectorExpression right) implements VectorExpression {
    public RowVectorConcat {
        if (!(left.columnVector() && right.columnVector())) {
            throw new IllegalArgumentException("Can only concatenate column vectors but found at least one row vector");
        }
    }

    @Override
    public int length() {
        return left.length() + right.length();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix leftEval = left.evaluate(bindings);
        final DMatrix rightEval = right.evaluate(bindings);
        final double[] values = new double[length()];
        for (int i = 0; i < leftEval.getNumRows(); i++) {
            values[i] = leftEval.get(i, 0);
        }
        for (int i = 0; i < rightEval.getNumRows(); i++) {
            values[leftEval.getNumRows() + i] = rightEval.get(i, 0);
        }

        return new DMatrixRMaj(length(), 1, true, values);
    }

    @Override
    public boolean columnVector() {
        return true;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);

        return new MatrixRowConcat(new DMatrixExpression(leftDerivative), new DMatrixExpression(rightDerivative)).evaluate(bindings);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final DMatrix leftPartial = left.computePartialDerivative(bindings, variable);
        final DMatrix rightPartial = right.computePartialDerivative(bindings, variable);

        return new RowVectorConcat(new DMatrixColumnVectorExpression(leftPartial), new DMatrixColumnVectorExpression(rightPartial)).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }
}
