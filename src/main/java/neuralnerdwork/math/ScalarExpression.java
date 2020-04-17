package neuralnerdwork.math;

public interface ScalarExpression {
    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The scalar value of this expression with all parameters substituted from the given bindings.
     */
    double evaluate(Model.Binder bindings);

    /**
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return An expression that is the partial derivative of this one, with respect to the given variable.
     */
    ScalarExpression computePartialDerivative(int variable);

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.Binder) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * This method has a default implementation that calls {@link #computePartialDerivative(int)} for each given
     * variable and combines them into the gradient. This may be inefficient for nested expressions.
     *
     * @param variables An ordered list of variables by which to differentiate this expression.
     * @return The gradient vector expression for this scalar expression, with respect to the given variables (and order).
     */
    default VectorExpression computeDerivative(int[] variables) {
        final ScalarExpression[] partialDerivatives = new ScalarExpression[variables.length];
        for (int i = 0; i < variables.length; i++) {
            partialDerivatives[i] = computePartialDerivative(variables[i]);
        }

        return new ScalarComponentsVector(partialDerivatives);
    }
}
