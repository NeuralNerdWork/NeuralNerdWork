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
    Vector evaluate(Model.Binder bindings);

    /**
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return An expression that is the partial derivative of this one, with respect to the given variable.
     */
    VectorExpression computePartialDerivative(int variable);

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.Binder) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * This method has a default implementation that calls {@link #computePartialDerivative(int)} for each given
     * variable and combines them into the derivative matrix. This may be inefficient for nested expressions.
     *
     * @param variables An ordered list of variables by which to differentiate this expression.
     * @return <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">The derivative matrix</a> expression
     * for this vector expression, with respect to the given variables (and order).
     */
    default MatrixExpression computeDerivative(int[] variables) {
        final VectorExpression[] columns = new VectorExpression[variables.length];
        for (int i = 0; i < variables.length; i++) {
            columns[i] = computePartialDerivative(variables[i]);
        }

        return new ColumnMatrix(columns);
    }
}
