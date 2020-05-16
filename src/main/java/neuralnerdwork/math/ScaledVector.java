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
    public Vector evaluate(Model.Binder bindings) {
        final Vector vector = vectorExpression.evaluate(bindings);
        final double value = scalarExpression.evaluate(bindings);
        final double[] scaled = new double[vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            scaled[i] = value * vector.get(i);
        }

        return new ConstantVector(scaled);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return MatrixSum.sum(
            new ScaledMatrix(scalarExpression, vectorExpression.computeDerivative(variables)),
            MatrixProduct.product(
                    new ColumnMatrix(vectorExpression),
                    // TODO Make RowMatrix
                    new Transpose(new ColumnMatrix(scalarExpression.computeDerivative(variables)))
            )
        );
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return VectorSum.sum(
                new ScaledVector(scalarExpression, vectorExpression.computePartialDerivative(variable)),
                new ScaledVector(scalarExpression.computePartialDerivative(variable), vectorExpression)
        );
    }

    @Override
    public boolean isZero() {
        return scalarExpression.isZero() || vectorExpression.isZero();
    }
}
