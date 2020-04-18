package neuralnerdwork.math;

public record SingleVariableExpression(int variable,
                                       ScalarExpression expression) implements SingleVariableFunction {
    @Override
    public String getFunctionName() {
        return "single variable function";
    }

    @Override
    public double apply(double input) {
        final Model.Binder binder = new Model.Binder(variable, 1);
        binder.put(variable, input);
        return expression.evaluate(binder);
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new SingleVariableExpression(variable, expression.computePartialDerivative(variable));
    }
}
