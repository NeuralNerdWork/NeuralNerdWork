package neuralnerdwork.math;

public interface ScalarExpression {
    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The scalar value of this expression with all parameters substituted from the given bindings.
     */
    double evaluate(Model.ParameterBindings bindings);

    /**
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return An expression that is the partial derivative of this one, with respect to the given variable.
     */
    ScalarExpression computePartialDerivative(int variable);

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.ParameterBindings) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * @param variables An ordered list of variables by which to differentiate this expression.
     * @return The gradient vector expression for this scalar expression, with respect to the given variables (and order).
     */
    VectorExpression computeDerivative(int[] variables);
}
