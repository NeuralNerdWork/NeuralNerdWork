package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public interface ScalarExpression {
    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The scalar value of this expression with all parameters substituted from the given bindings.
     */
    double evaluate(Model.ParameterBindings bindings);

    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @param variable The index of a variable by which this expression should be differentiated.
     * @return A scalar value that is the partial derivative of this one, with respect to the given variable at the given bindings.
     */
    double computePartialDerivative(Model.ParameterBindings bindings, int variable);

    /**
     * @return If this method returns true, the given expression is guaranteed to {@link #evaluate(Model.ParameterBindings) evaluate}
     * to zero, no matter what bindings it is given.
     */
    boolean isZero();

    /**
     * @param bindings A mapping of variable indices to values, used for substitution in this expression.
     * @return The gradient vector for this scalar expression, with respect to the given variables (and order) at the given bindings.
     */
    DMatrix computeDerivative(Model.ParameterBindings bindings);
}
