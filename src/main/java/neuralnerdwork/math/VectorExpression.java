package neuralnerdwork.math;

public interface VectorExpression {
    /**
     * @return Length of output vectorExpression
     */
    int length();

    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The vector value of this expression with all parameters substituted from the given bindings.
     */
    Vector evaluate(Model.ParameterBindings bindings);

    /**
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return An expression that is the partial derivative of this one, with respect to the given variable.
     */
    VectorExpression computePartialDerivative(int variable);

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.ParameterBindings) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * @param variables An ordered list of variables by which to differentiate this expression.
     * @return <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">The derivative matrix</a> expression
     * for this vector expression, with respect to the given variables (and order).
     */
    MatrixExpression computeDerivative(int[] variables);
}
