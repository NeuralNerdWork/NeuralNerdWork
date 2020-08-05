package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

/**
 * Does the weight update step for a variation of gradient descent. See
 * <a href="https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants">here</a>
 * for an overview of gradient descent variants.
 */
public interface WeightUpdateStrategy {

    /**
     * Updates weights based on current error and possibly (depending on implementation) internal state.
     *
     * @param error A differentiable expression for the current error of the model.
     * @param parameterBindings A map of parameters to values used in the error expression. Implementations MUST NOT
     *                          mutate this.
     * @return A vector that can be added to the current parameter bindings to update their values.
     */
    double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings);
}
