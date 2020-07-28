package neuralnerdwork.backprop;

import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

import java.util.Objects;

import static neuralnerdwork.math.MatrixProduct.product;

public record FeedForwardNetwork(Layer<?>[]layers) {

    private static class StatefulLayerDelegate<C> {
        private final Layer<C> layer;
        private C cache;

        private StatefulLayerDelegate(Layer<C> layer) {
            this.layer = layer;
        }

        DMatrix derivativeWithRespectToLayerInput(Vector layerInput, Model.ParameterBindings bindings) {
            var result = layer.derivativeWithRespectToLayerInput(layerInput, Objects.requireNonNull(cache), bindings);
            cache = result.cache();

            return result.output();
        }

        Vector derivativeWithRespectLayerParameter(Vector layerInput, int variable, Model.ParameterBindings bindings) {
            var result = layer
                    .derivativeWithRespectLayerParameter(layerInput, variable, Objects.requireNonNull(cache), bindings);
            cache = result.cache();

            return result.output();
        }

        Vector evaluate(Vector layerInput, Model.ParameterBindings bindings) {
            var result = layer.evaluate(layerInput, bindings);
            cache = result.cache();

            return result.output();
        }

        Vector getCachedEvaluation() {
            return layer.getEvaluation(Objects.requireNonNull(cache));
        }
    }

    public int inputLength() {
        return layers[0].inputLength();
    }

    public int outputLength() {
        return layers[layers.length - 1].outputLength();
    }

    public FeedForwardExpression expression(ConstantVector input) {
        return new FeedForwardExpression(layers, input);
    }

    public record FeedForwardExpression(Layer<?>[]layers, ConstantVector input) implements VectorExpression {
        @Override
        public int length() {
            return layers[layers.length - 1].outputLength();
        }

        @Override
        public Vector evaluate(Model.ParameterBindings bindings) {
            Vector lastOutput = input;
            for (Layer<?> layer : layers) {
                lastOutput = layer.evaluate(lastOutput, bindings).output();
            }

            return lastOutput;
        }


        @Override
        public DMatrix computeDerivative(Model.ParameterBindings bindings) {
            /*
             Algorithm Summary:
                - (Bottom to top) Evaluate network with current parameter arguments, saving activations and weighted sums for each layer
                - (Top to bottom) Using pre-calculated activations and weighted sums, calculate error deltas at each layer
                        Note: Lower layers error deltas depend on higher layers
                - Calculate partial derivatives for each parameter using deltas
                        Note: We do this as a separate step because of the current VectorExpression API requiring
                                partial derivatives to be returned in a layout consistent with a given variable order.
             */

            Vector lastOutput = input;

            /* Feed forward
                Evaluate network, saving activation values and weighted sums of inputs at each layer to be re-used
                in derivative calculations.
             */
            var layerDelegates = new StatefulLayerDelegate<?>[layers.length];
            for (int l = 0; l < layers.length; l++) {
                final Layer<?> layer = layers[l];
                var delegate = new StatefulLayerDelegate<>(layer);
                layerDelegates[l] = delegate;

                // warms cache
                lastOutput = delegate.evaluate(lastOutput, bindings);
            }

            /* Backpropogate
                Calculate deltas starting at last layer, going backwards.
                This code has some extra complexity because the input and output layers are both special cases.
                The input layer is not represented in the `layers` array, and the output layer does not have bias.

                The formula for deltas at each layer is recursive, based on definitions of deltas of higher layers.
             */
            final DMatrix[] deltas = new DMatrix[layers.length];
            // Base case: output layer
            {
                final int layerIndex = layers.length - 1;
                final Vector layerInput = (layerIndex != 0) ?
                        layerDelegates[layerIndex - 1].getCachedEvaluation() :
                        input;
                deltas[layerIndex] = layerDelegates[layerIndex].derivativeWithRespectToLayerInput(layerInput, bindings);
            }

            for (int l = layers.length - 2; l > -1; l--) {
                final StatefulLayerDelegate<?> delegate = layerDelegates[l];
                try {
                    final DMatrix previousDeltaExpression = deltas[l + 1];

                    final Vector layerInput = (l == 0) ? input : layerDelegates[l - 1].getCachedEvaluation();
                    final DMatrix layerDerivative = delegate.derivativeWithRespectToLayerInput(layerInput, bindings);
                    deltas[l] =
                            product(
                                    new DMatrixExpression(previousDeltaExpression),
                                    new DMatrixExpression(layerDerivative)
                            ).evaluate(bindings);
                } catch (IllegalArgumentException iae) {
                    throw new RuntimeException("Problem building deltas in layer index " + l + " of " + layers.length, iae);
                }
            }

            /* Build up derivatives using deltas and activations
                We do this separately from the last step so that we can iterate in order that
                variables were listed in `computeDerivative`.
             */
            final double[][] partialDerivatives = new double[layers[layers.length - 1].outputLength()][bindings.size()];
            int varIndex = 0;
            for (int variable : bindings.variables()) {
                final int layerIndex = findLayerIndex(variable);
                final StatefulLayerDelegate<?> delegate = layerDelegates[layerIndex];

                try {
                    /*
                     Special cases for input layer
                     */
                    final Vector layerInput;
                    if (layerIndex > 0) {
                        layerInput = layerDelegates[layerIndex - 1].getCachedEvaluation();
                    } else {
                        layerInput = input;
                    }

                    final Vector layerDerivative = delegate
                            .derivativeWithRespectLayerParameter(layerInput, variable, bindings);
                    if (layerIndex == layers.length - 1) {
                        for (int i = 0; i < layerDerivative.length(); i++) {
                            partialDerivatives[i][varIndex] = layerDerivative.get(i);
                        }
                    } else {
                        final DMatrix layerDeltas = deltas[layerIndex + 1];
                        Vector derivativeProduct = new MatrixVectorProduct(
                                new DMatrixExpression(layerDeltas),
                                layerDerivative
                        ).evaluate(bindings);
                        for (int i = 0; i < derivativeProduct.length(); i++) {
                            partialDerivatives[i][varIndex] = derivativeProduct.get(i);
                        }
                    }
                    varIndex++;
                } catch (RuntimeException e) {
                    throw new RuntimeException(String.format("Problem in (varIndex/totalVars, layerIndex/totalLayers) = (%d/%d, %d/%d)",
                                                             varIndex, bindings.size(), layerIndex, layers.length), e);
                }
            }

            return new DMatrixRMaj(partialDerivatives);
        }

        @Override
        public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override
        public boolean isZero() {
            return false;
        }

        private int findLayerIndex(int variable) {
            for (int l = 0; l < layers.length; l++) {
                if (layers[l].containsVariable(variable)) {
                    return l;
                }
            }

            throw new IllegalArgumentException("Cannot find layer containing variable " + variable);
        }

    }
}
