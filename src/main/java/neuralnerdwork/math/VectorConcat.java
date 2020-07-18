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
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        final MatrixExpression leftDerivative = left.computeDerivative(bindings, variables);
        final MatrixExpression rightDerivative = right.computeDerivative(bindings, variables);

        return new MatrixRowConcat(leftDerivative, rightDerivative).evaluate(bindings);
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
