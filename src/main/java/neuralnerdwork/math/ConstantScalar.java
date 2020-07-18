package neuralnerdwork.math;

public record ConstantScalar(double value) implements ScalarExpression {
    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return value;
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return 0.0;
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        return new ConstantVector(new double[variables.length]);
    }

    @Override
    public boolean isZero() {
        return value == 0.0;
    }
}
