package neuralnerdwork.math;

public record SingleVariableExpression(int variable,
                                       ScalarExpression expression) implements SingleVariableFunction {
    @Override
    public String getFunctionName() {
        return "single variable function";
    }

    @Override
    public double apply(double input) {
        final Model.ParameterBindings parameterBindings = new Model.ParameterBindings(variable, 1);
        parameterBindings.put(variable, input);
        return expression.evaluate(parameterBindings);
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new SingleVariableExpression(variable, expression.computePartialDerivative(variable));
    }
}
