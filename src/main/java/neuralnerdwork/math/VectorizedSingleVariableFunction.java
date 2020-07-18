package neuralnerdwork.math;

public record VectorizedSingleVariableFunction(SingleVariableFunction function, VectorExpression vectorExpression) implements VectorExpression {

    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        final Vector vector = vectorExpression.evaluate(bindings);
        final double[] values = new double[vector.length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = function.apply(vector.get(i));
        }

        return new ConstantVector(values);
    }

    @Override
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        final MatrixExpression outerDerivative = new DiagonalizedVector(
                new VectorizedSingleVariableFunction(
                        function.differentiateByInput(),
                        vectorExpression
                )
        );

        return MatrixProduct.product(outerDerivative, vectorExpression.computeDerivative(bindings, variables))
                            .evaluate(bindings);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final VectorExpression innerDerivative = this.vectorExpression.computePartialDerivative(bindings, variable);

        return VectorComponentProduct.product(
                new VectorizedSingleVariableFunction(function.differentiateByInput(),
                                                     vectorExpression),
                innerDerivative).evaluate(bindings);
    }
}
