package neuralnerdwork.math;

public record ScaledVector(double value, VectorExpression vectorExpression) implements VectorExpression {
    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final Vector vector = vectorExpression.evaluate(bindings);
        final double[] scaled = new double[vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            scaled[i] = value * vector.get(i);
        }

        return new ConstantVector(scaled);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return new ScaledMatrix(value, vectorExpression.computeDerivative(variables));
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return new ScaledVector(value, vectorExpression.computePartialDerivative(variable));
    }

    @Override
    public boolean isZero() {
        return value == 0.0 || vectorExpression.isZero();
    }
}
