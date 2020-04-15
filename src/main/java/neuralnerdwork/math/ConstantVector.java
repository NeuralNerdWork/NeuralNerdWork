package neuralnerdwork.math;

public record ConstantVector(double[] values) implements VectorFunction, Vector {

    @Override
    public double get(int index) {
        return values[index];
    }

    @Override
    public int length() {
        return values.length;
    }

    @Override
    public int inputLength() {
        return 0;
    }

    @Override
    public Vector apply(double[] input) {
        return this;
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        return new ConstantVector(new double[values.length]);
    }
}
