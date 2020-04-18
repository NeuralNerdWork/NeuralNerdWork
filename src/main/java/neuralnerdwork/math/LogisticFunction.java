package neuralnerdwork.math;

public class LogisticFunction implements SingleVariableFunction {
    @Override
    public String getFunctionName() {
        return "logistic";
    }

    @Override
    public double apply(double input) {
        final double exp = Math.exp(-input);
        if (Double.isInfinite(exp)) {
            return 0;
        } else {
            return 1 / (1 + exp);
        }
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        final int variable = 0;
        final ScalarParameter param = new ScalarParameter(variable);
        final ScalarExpression derivativeExpression = ScalarProduct.product(this.invoke(param), this.invoke(new ScalarConstantMultiple(-1.0, param)));

        return new SingleVariableExpression(variable, derivativeExpression);
    }
}
