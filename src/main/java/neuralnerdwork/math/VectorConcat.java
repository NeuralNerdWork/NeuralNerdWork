package neuralnerdwork.math;

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
    public MatrixExpression computeDerivative(int[] variables) {
        final MatrixExpression leftDerivative = left.computeDerivative(variables);
        final MatrixExpression rightDerivative = right.computeDerivative(variables);

        return new MatrixRowConcat(leftDerivative, rightDerivative);
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        final VectorExpression leftPartial = left.computePartialDerivative(variable);
        final VectorExpression rightPartial = right.computePartialDerivative(variable);

        return new VectorConcat(leftPartial, rightPartial);
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }
}
