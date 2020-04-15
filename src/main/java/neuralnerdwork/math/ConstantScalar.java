package neuralnerdwork.math;

public record ConstantScalar(double value) implements SingleVariableFunction {

    @Override
    public double apply(double[] input) {
        return value;
    }

    @Override
    public double apply(double input) {
        return value;
    }

    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        return new ConstantScalar(0.0);
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        return new ConstantScalar(0.0);
    }
}
