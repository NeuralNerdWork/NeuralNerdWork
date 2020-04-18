package neuralnerdwork.math;

public class SquaredSingleVariableFunction implements SingleVariableFunction {
    @Override
    public String getFunctionName() {
        return "sqaured";
    }

    @Override
    public double apply(double input) {
        return input * input;
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        final int var = 0;
        final ScalarExpression expression = new ScalarConstantMultiple(2.0, new ScalarParameter(var));

        return new SingleVariableExpression(var, expression);
    }
}
