package neuralnerdwork.math;

public class NegateScalar implements ScalarFunction {
    @Override
    public double apply(double[] input) {
        if (input.length != 1) {
            throw new IllegalArgumentException("negation only works on scalar input");
        }

        return -input[0];
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        if (variableIndex != 0) {
            throw new IndexOutOfBoundsException(variableIndex);
        }

        return new ConstantScalar(-1.0);
    }
}
