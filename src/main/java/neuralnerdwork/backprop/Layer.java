package neuralnerdwork.backprop;

import neuralnerdwork.math.ActivationFunction;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.Vector;
import org.ejml.data.DMatrix;

import java.util.stream.IntStream;

/**
 * A layer of a neural network. All methods should be referentially transparent.
 *
 * @param <C> A cache object for reusing intermediate results across multiple method calls.
 */
public interface Layer<C> {

    ActivationFunction activation();

    /**
     * Tuple for returning computation result and updated cache object.
     *
     * @param <O> Result type.
     * @param <C> Cache type.
     */
    record Result<O, C>(O output, C cache) {}

    /**
     * @param variable A variable in the network.
     * @return true iff this layer contains the given variable.
     */
    boolean containsVariable(int variable);

    /**
     * @return The length of the vector output by this layer.
     */
    int outputLength();

    /**
     * @return The length of the input vector to this layer.
     */
    int inputLength();

    /**
     * @return The exhaustive set of variables in this layer.
     */
    IntStream variables();

    /**
     * Used for back-propogating derivative to lower layers.
     * This method is always called after {@link #evaluate(Vector, Model.ParameterBindings)}.
     *
     * @param layerInput The input vector for this layer. Never null.
     * @param cache Cache object that may contain intermediate results. Never null. Fields in cache maybe null.
     * @param bindings Bindings of parameters in entire network. Never null.
     * @return The derivative of this layer with respect to the input of this layer.
     */
    Result<DMatrix, C> derivativeWithRespectToLayerInput(Vector layerInput, C cache, Model.ParameterBindings bindings);

    /**
     * Used for back-propogating derivative to lower layers.
     * This method is always called after {@link #evaluate(Vector, Model.ParameterBindings)}.
     *
     * @param layerInput The input vector for this layer. Never null.
     * @param variable A variable that is guaranteed to give result {@code true} as an argument to {@link #containsVariable(int)}.
     * @param cache Cache object that may contain intermediate results. Never null. Fields in cache maybe null.
     * @param bindings Bindings of parameters in entire network. Never null.
     * @return The derivative of this layer with respect to a variable in this layer.
     */
    Result<Vector, C> derivativeWithRespectLayerParameter(Vector layerInput, int variable, C cache, Model.ParameterBindings bindings);

    /**
     * Evaluate this layer given an input vector.
     *
     * @param layerInput The input vector to this layer (the output of the previous layer, or the input to the whole network for the first hidden layer).
     * @param bindings Bindings of parameters in entire network. Never null.
     * @return The output of this layer, given a particular input vector and parameter bindings.
     */
    Result<Vector, C> evaluate(Vector layerInput, Model.ParameterBindings bindings);

    /**
     * @param cache A cache object from a previous call to {@link #evaluate(Vector, Model.ParameterBindings)}. Must not be null.
     * @return The cached evaluation result.
     */
    Vector getEvaluation(C cache);
}
