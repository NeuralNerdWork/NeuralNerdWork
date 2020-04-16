package neuralnerdwork.math;

public record SingleVariableLogisticFunction(String name) implements SingleVariableFunction {
    public SingleVariableLogisticFunction() {
        this("logistic");
    }

    @Override
    public double apply(double input) {
        final double eToX = Math.exp(input);
        return eToX / (1 + eToX);
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        final int variable = 0;
        final ScalarParameter param = new ScalarParameter(variable);
        final ScalarExpression derivativeExpression = ScalarProduct.product(this.invoke(param), this.invoke(new ScalarConstantMultiple(-1.0, param)));

        return new SingleVariableExpression(variable, derivativeExpression);
    }
}
