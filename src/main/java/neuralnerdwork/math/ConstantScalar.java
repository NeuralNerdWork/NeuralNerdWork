package neuralnerdwork.math;

public record ConstantScalar(double value) implements ScalarFunction {

    @Override
    public double apply(double[] input) {
        return value;
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        return new ConstantScalar(0.0);
    }
}
