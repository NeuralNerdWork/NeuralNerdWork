package neuralnerdwork.math;

public class NegateScalar implements SingleVariableFunction {
    @Override
    public double apply(double input) {
        return -input;
    }

    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        return new ConstantScalar(-1.0);
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        if (variableIndex != 0) {
            throw new IndexOutOfBoundsException(variableIndex);
        }

        return new ConstantScalar(-1.0);
    }
}
