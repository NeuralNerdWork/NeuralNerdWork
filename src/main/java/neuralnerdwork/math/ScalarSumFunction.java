package neuralnerdwork.math;

public record ScalarSumFunction(ScalarFunction left, ScalarFunction right) implements ScalarFunction {

    @Override
    public double apply(double[] input) {
        return left.apply(input) + right.apply(input);
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        return new ScalarSumFunction(left.differentiate(variableIndex), right.differentiate(variableIndex));
    }
}
