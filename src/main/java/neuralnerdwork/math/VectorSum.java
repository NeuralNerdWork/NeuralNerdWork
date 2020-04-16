package neuralnerdwork.math;

public record VectorSum(VectorExpression left, VectorExpression right) implements VectorExpression {
    public static VectorExpression sum(VectorExpression left, VectorExpression right) {
        final VectorSum sum = new VectorSum(left, right);
        if (sum.isZero()) {
            return new ConstantVector(new double[sum.length()]);
        } else if (left.isZero()) {
            return right;
        } else if (right.isZero()) {
            return left;
        } else {
            return sum;
        }
    }

    public VectorSum {
        if (left.length() != right.length()) {
            throw new IllegalArgumentException("Cannot add vectors of different lengths");
        }
    }

    @Override
    public int length() {
        return left.length();
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final Vector left = this.left.evaluate(bindings);
        final Vector right = this.right.evaluate(bindings);
        final int length = Math.max(left.length(), right.length());

        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = left.get(i) + right.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return MatrixSum.sum(left.computeDerivative(variables), right.computeDerivative(variables));
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return VectorSum.sum(
                this.left.computePartialDerivative(variable),
                this.right.computePartialDerivative(variable)
        );
    }
}
