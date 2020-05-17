package neuralnerdwork.math;

public record ZeroPaddedVector(VectorExpression expression, int zeros) implements VectorExpression {
    @Override
    public int length() {
        return expression.length() + zeros;
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final Vector vector = expression.evaluate(bindings);
        return new Vector() {
            @Override
            public double get(int index) {
                if (index < vector.length()) {
                    return vector.get(index);
                } else if (index < ZeroPaddedVector.this.length()) {
                    return 0.0;
                } else {
                    throw new IllegalArgumentException(String.format("Invalid index %d in vector of length %d", index, length()));
                }
            }

            @Override
            public int length() {
                return ZeroPaddedVector.this.length();
            }
        };
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        throw new RuntimeException("Not yet implemented!");
    }

    @Override
    public boolean isZero() {
        return expression.isZero();
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        throw new RuntimeException("Not yet implemented!");
    }
}
