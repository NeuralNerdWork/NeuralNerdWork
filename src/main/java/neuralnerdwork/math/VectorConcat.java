package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public record VectorConcat(VectorExpression left, VectorExpression right) implements VectorExpression {
    @Override
    public int length() {
        return left.length() + right.length();
    }

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        final Vector leftEval = left.evaluate(bindings);
        final Vector rightEval = right.evaluate(bindings);
        final double[] values = new double[length()];
        for (int i = 0; i < leftEval.length(); i++) {
            values[i] = leftEval.get(i);
        }
        for (int i = 0; i < rightEval.length(); i++) {
            values[leftEval.length() + i] = rightEval.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);

        return new MatrixRowConcat(new DMatrixExpression(leftDerivative), new DMatrixExpression(rightDerivative)).evaluate(bindings);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final VectorExpression leftPartial = left.computePartialDerivative(bindings, variable);
        final VectorExpression rightPartial = right.computePartialDerivative(bindings, variable);

        return new VectorConcat(leftPartial, rightPartial).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }
}
