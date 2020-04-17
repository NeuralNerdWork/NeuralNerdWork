package neuralnerdwork.math;

public record SquaredSingleVariableFunction() implements SingleVariableFunction {
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
