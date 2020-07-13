package neuralnerdwork.math;

import java.util.Optional;

import static neuralnerdwork.math.MatrixProduct.product;

public record FeedForwardNetwork(Layer[] layers) {
    public record Layer(ParameterMatrix weights, Optional<ParameterVector> bias, SingleVariableFunction activation) {}

    public FeedForwardExpression expression(ConstantVector input) {
        return new FeedForwardExpression(layers, input);
    }

    public record FeedForwardExpression(Layer[] layers, ConstantVector input) implements VectorExpression {
        @Override
        public int length() {
            return layers[layers.length-1].weights().rows();
        }

        @Override
        public Vector evaluate(Model.ParameterBindings bindings) {
            VectorExpression network = input;
            for (int l = 0; l < layers.length; l++) {
                final Layer layer = layers[l];
                final VectorExpression layerActivation = weightedSumExpression(network, layer);
                network = new VectorizedSingleVariableFunction(
                        layer.activation(),
                        layerActivation
                );
            }

            return network.evaluate(bindings);
        }

        private static VectorExpression weightedSumExpression(VectorExpression input, Layer layer) {
            final var weightedSums = new MatrixVectorProduct(
                    layer.weights(),
                    input
            );

            return layer.bias()
                        .map(b -> VectorSum.sum(weightedSums, b))
                        .orElse(weightedSums);
        }


        @Override
        public MatrixExpression computeDerivative(int[] variables) {
            record LayerEval(double[] activations, double[] activationInputs) {}

            // FIXME do something with variables?
            return new MatrixExpression() {
                @Override
                public int rows() {
                    return length();
                }

                @Override
                public int cols() {
                    return variables.length;
                }

                @Override
                public Matrix evaluate(Model.ParameterBindings bindings) {
                /*
                 Algorithm Summary:
                    - (Bottom to top) Evaluate network with current parameter arguments, saving activations and weighted sums for each layer
                    - (Top to bottom) Using pre-calculated activations and weighted sums, calculate error deltas at each layer
                            Note: Lower layers error deltas depend on higher layers
                    - Calculate partial derivatives for each parameter using deltas
                            Note: We do this as a separate step because of the current VectorExpression API requiring
                                    partial derivatives to be returned in a layout consistent with a given variable order.
                 */

                    final LayerEval[] feedForwardEvaluations = new LayerEval[layers.length];
                    double[] lastOutput = input.toArray();

                /* Feed forward
                    Evaluate network, saving activation values and weighted sums of inputs at each layer to be re-used
                    in derivative calculations.
                 */
                    for (int l = 0; l < layers.length; l++) {
                        final Layer layer = layers[l];
                        final var input = new ConstantVector(lastOutput);
                        final double[] weightedSums = weightedSumExpression(input, layer)
                                .evaluate(bindings)
                                .toArray();

                        lastOutput = new VectorizedSingleVariableFunction(
                                layer.activation(),
                                new ConstantVector(weightedSums)
                        ).evaluate(bindings)
                         .toArray();

                        feedForwardEvaluations[l] = new LayerEval(lastOutput, weightedSums);
                    }

                /* Backpropogate
                    Calculate deltas starting at last layer, going backwards.
                    This code has some extra complexity because the input and output layers are both special cases.
                    The input layer is not represented in the `layers` array, and the output layer does not have bias.

                    The formula for deltas at each layer is recursive, based on definitions of deltas of higher layers.
                 */
                    final Matrix[] deltas = new Matrix[layers.length];

                    // Special case: output layer
                    final Layer outputLayer = layers[layers.length - 1];
                    final SingleVariableFunction outputActivationDerivative = outputLayer.activation().differentiateByInput();
                    deltas[layers.length - 1] =
                            new DiagonalizedVector(
                                    new VectorizedSingleVariableFunction(
                                            outputActivationDerivative,
                                            new ConstantVector(feedForwardEvaluations[layers.length - 1].activationInputs())
                                    )
                            ).evaluate(bindings);

                    for (int l = layers.length-2; l > -1; l--) {
                        final Layer curLayer = layers[l];
                        final SingleVariableFunction curActivationDerivative = curLayer.activation().differentiateByInput();
                        try {
                        /*
                         If you have vectors x and y, then
                           x dot y == D(x) * y
                         where `dot` is the vector dot product, `*` is matrix multiplication, and `D` is a function
                         that turns a vector into a diagonal matrix.

                         Why does this matter? Matrix multiplication is associative, so for complex expressions we have:
                           x dot (A * y) == D(X) * (A * y) = (D(X) * A) * y

                         We use this in the backpropogation so that we can build up delta terms from left to right.
                         */

                            final MatrixExpression diagonalActivationDerivative =
                                    new DiagonalizedVector(
                                            new VectorizedSingleVariableFunction(
                                                    curActivationDerivative,
                                                    new ConstantVector(feedForwardEvaluations[l].activationInputs())
                                            )
                                    );

                            final MatrixExpression previousDeltaExpression = new ConstantMatrixExpression(deltas[l + 1]);

                            deltas[l] =
                                    product(
                                            previousDeltaExpression,
                                            product(
                                                    layers[l + 1].weights(),
                                                    diagonalActivationDerivative
                                            )
                                    ).evaluate(bindings);
                        } catch (IllegalArgumentException iae) {
                            throw new RuntimeException("Problem building deltas in layer index " + l + " of " + layers.length, iae);
                        }
                    }

                /* Build up derivatives using deltas and activations
                    We do this separately from the last step so that we can iterate in order that
                    variables were listed in `computeDerivative`.
                 */
                    final double[][] partialDerivatives = new double[variables.length][];
                    for (int varIndex = 0; varIndex < variables.length; varIndex++) {
                        final int variable = variables[varIndex];
                        final int layerIndex = findLayerIndex(variable);

                        try {
                            final double[] lowerLayerValues = new double[layers[layerIndex].weights().rows()];
                        /*
                         Special cases for input layer, and for bias weight
                         */
                            if (layerIndex > 0) {
                                if (layers[layerIndex].weights().containsVariable(variable)) {
                                    final int row = layers[layerIndex].weights().rowIndexFor(variable);
                                    final int col = layers[layerIndex].weights().colIndexFor(variable);
                                    lowerLayerValues[row] = feedForwardEvaluations[layerIndex - 1].activations()[col];
                                } else {
                                    int biasIndex = layers[layerIndex].bias()
                                                                      .filter(b -> b.containsVariable(variable))
                                                                      .map(b -> b.indexFor(variable))
                                                                      .orElseThrow(() -> new IllegalStateException("Cannot find variable " + variable + " in weights or bias"));
                                    lowerLayerValues[biasIndex] = 1.0;
                                }
                            } else {
                                if (layers[layerIndex].weights().containsVariable(variable)) {
                                    final int row = layers[layerIndex].weights().rowIndexFor(variable);
                                    final int col = layers[layerIndex].weights().colIndexFor(variable);
                                    lowerLayerValues[row] = input.get(col);
                                } else {
                                    int biasIndex = layers[layerIndex].bias()
                                                                      .filter(b -> b.containsVariable(variable))
                                                                      .map(b -> b.indexFor(variable))
                                                                      .orElseThrow(() -> new IllegalStateException("Cannot find variable " + variable + " in weights or bias"));
                                    lowerLayerValues[biasIndex] = 1.0;
                                }
                            }
                            final ConstantMatrixExpression layerDeltas = new ConstantMatrixExpression(deltas[layerIndex]);
                            partialDerivatives[varIndex] =
                                    new MatrixVectorProduct(
                                            layerDeltas,
                                            new ConstantVector(lowerLayerValues)
                                    ).evaluate(bindings)
                                     .toArray();
                        } catch (RuntimeException e) {
                            throw new RuntimeException(String.format("Problem in (varIndex/totalVars, layerIndex/totalLayers) = (%d/%d, %d/%d)",
                                                                     varIndex, variables.length, layerIndex, layers.length), e);
                        }
                    }

                    /*
                     * Because we represent matrices as row-major 2d-arrays, it is more work to assign an entire column
                     * at once than to assign an entire row. For convenience, we build up the transpose of the result.
                     */
                    return new Transpose(new ConstantArrayMatrix(partialDerivatives));
                }

                private int findLayerIndex(int variable) {
                    int l;
                    for (l = 0; l < layers.length; l++) {
                        Layer layer = layers[l];
                        if (layer.weights().containsVariable(variable) || layer.bias().filter(b -> b.containsVariable(variable)).isPresent()) {
                            break;
                        }
                    }
                    if (l == layers.length) {
                        throw new IllegalArgumentException("Cannot find layer containing variable " + variable);
                    }
                    return l;
                }

                @Override
                public boolean isZero() {
                    return false;
                }

                @Override
                public MatrixExpression computePartialDerivative(int variable) {
                    throw new RuntimeException("Not yet implemented!");
                }
            };
        }

        @Override
        public VectorExpression computePartialDerivative(int variable) {
            throw new RuntimeException("Not yet implemented!");
        }

        @Override
        public boolean isZero() {
            return false;
        }
    }
}
