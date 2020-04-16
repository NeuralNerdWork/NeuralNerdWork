package neuralnerdwork.math;

public interface ScalarExpression {
    double evaluate(Model.Binder bindings);
    ScalarExpression computePartialDerivative(int variable);
    boolean isZero();

    default VectorExpression computeDerivative(int[] variables) {
        final ScalarExpression[] partialDerivatives = new ScalarExpression[variables.length];
        for (int i = 0; i < variables.length; i++) {
            partialDerivatives[i] = computePartialDerivative(variables[i]);
        }

        return new ScalarComponentsVector(partialDerivatives);
    }
}
