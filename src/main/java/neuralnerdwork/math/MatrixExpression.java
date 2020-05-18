package neuralnerdwork.math;

public interface MatrixExpression {
    /**
     * @return The number of rows in the {@link #evaluate(Model.ParameterBindings) evaluated} matrix.
     */
    int rows();

    /**
     * @return The number of columns in the {@link #evaluate(Model.ParameterBindings) evaluated} matrix.
     */
    int cols();

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.ParameterBindings) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The matrix value of this expression with all parameters substituted from the given bindings.
     */
    Matrix evaluate(Model.ParameterBindings bindings);

    /**
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return An expression that is the partial derivative of this one, with respect to the given variable.
     */
    MatrixExpression computePartialDerivative(int variable);
}
