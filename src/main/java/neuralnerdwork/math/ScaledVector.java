package neuralnerdwork.math;

public record ScaledVector(ScalarExpression scalarExpression, VectorExpression vectorExpression) implements VectorExpression {
    public ScaledVector(double scalar, VectorExpression vectorExpression) {
        this(new ConstantScalar(scalar), vectorExpression);
    }

    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        final Vector vector = vectorExpression.evaluate(bindings);
        final double value = scalarExpression.evaluate(bindings);
        final double[] scaled = new double[vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            scaled[i] = value * vector.get(i);
        }

        return new ConstantVector(scaled);
    }

    @Override
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        return MatrixSum.sum(
            new ScaledMatrix(scalarExpression, vectorExpression.computeDerivative(bindings, variables)),
            MatrixProduct.product(
                    new ColumnMatrix(vectorExpression),
                    // TODO Make RowMatrix
                    new TransposeExpression(new ColumnMatrix(scalarExpression.computeDerivative(bindings, variables)))
            )
        ).evaluate(bindings);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                new ScaledVector(scalarExpression, vectorExpression.computePartialDerivative(bindings, variable)),
                new ScaledVector(scalarExpression.computePartialDerivative(bindings, variable), vectorExpression)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return scalarExpression.isZero() || vectorExpression.isZero();
    }
}
