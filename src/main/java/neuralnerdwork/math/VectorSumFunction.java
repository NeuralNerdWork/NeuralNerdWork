package neuralnerdwork.math;

public record VectorSumFunction(VectorFunction left, VectorFunction right) implements VectorFunction {
    @Override
    public Vector apply(double[] input) {
        final Vector left = this.left.apply(input);
        final Vector right = this.right.apply(input);
        final int length = Math.max(left.length(), right.length());

        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = left.get(i) + right.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        return new VectorSumFunction(left.differentiate(variableIndex), right.differentiate(variableIndex));
    }
}
