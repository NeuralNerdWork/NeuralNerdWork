package neuralnerdwork.descent;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

import java.util.List;
import java.util.function.Function;

/**
 * Runs a variation of gradient descent. See <a href="https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants">here</a>
 * for an overview of gradient descent variants.
 */
public interface GradientDescentStrategy {

    /**
     * Runs a variation of gradient descent.
     *
     * @param trainingSamples A list of labelled training data.
     * @param parameterBindings A map of parameter variables to values. Values are updated as part of the gradient descent process.
     * @param errorFunction A function yielding an expression for the error of a model on given training samples based on current
     *                      parameter values in the given {@link neuralnerdwork.math.Model.ParameterBindings}. This should not have side-effects.
     * @return The updated {@link neuralnerdwork.math.Model.ParameterBindings} after running gradient descent to termination.
     * Termination conditions are at the discretion of implementations.
     */
    Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples, Model.ParameterBindings parameterBindings, Function<List<TrainingSample>, ScalarExpression> errorFunction, TerminationPredicate terminationPredicate);
}
