package neuralnerdwork.math;

public record FeedForwardNetwork(ConstantVector input, Layer[] layers) implements VectorExpression {
    public record Layer(ParameterMatrix weights, SingleVariableFunction activation) {}
    @Override
    public int length() {
        return layers[layers.length-1].weights().rows();
    }

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        VectorExpression network = input;
        final ConstantVector biasComponent = new ConstantVector(new double[]{1.0});
        for (int l = 0; l < layers.length; l++) {
            final Layer layer = layers[l];
            network = new VectorizedSingleVariableFunction(
                    layer.activation(),
                    new MatrixVectorProduct(
                            layer.weights(),
                            new VectorConcat(network, biasComponent)
                    )
            );
        }

        return network.evaluate(bindings);
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
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

                record LayerEval(double[] activations, double[] activationInputs) {}
                final LayerEval[] feedForwardEvaluations = new LayerEval[layers.length];
                final ConstantVector biasComponent = new ConstantVector(new double[]{1.0});
                double[] lastOutput = input.toArray();

                /* Feed forward
                    Evaluate network, saving activation values and weighted sums of inputs at each layer to be re-used
                    in derivative calculations.
                 */
                for (int l = 0; l < layers.length; l++) {
                    final Layer layer = layers[l];
                    final VectorConcat input = new VectorConcat(new ConstantVector(lastOutput), biasComponent);
                    final double[] curActivationInputs =
                            new MatrixVectorProduct(
                                    layer.weights(),
                                    input

                            ).evaluate(bindings)
                             .toArray();

                    lastOutput = new VectorizedSingleVariableFunction(
                            layer.activation(),
                            new ConstantVector(curActivationInputs)
                    ).evaluate(bindings)
                     .toArray();

                    feedForwardEvaluations[l] = new LayerEval(lastOutput, curActivationInputs);
                }

                /* Backpropogate
                    Calculate deltas starting at last layer, going backwards.
                    This code has some extra complexity because the input and output layers are both special cases.
                    The input layer is not represented in the `layers` array, and the ouptut layer does not have bias.

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
                         The way we handle bias is by padding the activation of a previous layer with a `1`.
                         Whenever we take a derivative of a 1-padded vector, the padding turns into a zero.
                         */
                        final MatrixExpression paddedDiagonalActivationDerivative =
                                new ZeroPaddedMatrix(
                                        new DiagonalizedVector(
                                                new VectorizedSingleVariableFunction(
                                                        curActivationDerivative,
                                                        new ConstantVector(feedForwardEvaluations[l].activationInputs())
                                                )
                                        ),
                                        1,
                                        1
                                );
                        /*
                         The right-most column of the previous delta are errors associated with
                         the bias of that layer. Since the bias is added at each layer, we need to strip
                         off that column so the matrix multiplication is valid.
                         */
                        final MatrixExpression truncatedUpperLayerDelta =
                                new TruncatedMatrix(
                                        new ConstantMatrixExpression(deltas[l + 1]),
                                        deltas[l + 1].rows(),
                                        // don't need to strip off anything for output layer that has no bias
                                        deltas[l + 1].cols() - (l == layers.length - 2 ? 0 : 1)
                                );
                        deltas[l] =
                                new MatrixProduct(
                                        new MatrixProduct(
                                                truncatedUpperLayerDelta,
                                                layers[l + 1].weights()
                                        ),
                                        paddedDiagonalActivationDerivative
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
                    final int row = layers[layerIndex].weights().rowIndexFor(variable);
                    final int col = layers[layerIndex].weights().colIndexFor(variable);

                    try {
                        final double[] lowerLayerValues = (layerIndex == layers.length - 1) ?
                                                          new double[layers[layerIndex].weights().rows()] :
                                                          // +1 for bias added to hidden layer output
                                                          new double[layers[layerIndex].weights().rows() + 1];
                        /*
                         Special cases for input layer, and for bias weight
                         (which has a `1.0` as input instead of an activation from previous layer)
                         */
                        if (layerIndex > 0) {
                            if (col < feedForwardEvaluations[layerIndex - 1].activations().length) {
                                lowerLayerValues[row] = feedForwardEvaluations[layerIndex - 1].activations()[col];
                            } else {
                                lowerLayerValues[row] = 1.0;
                            }
                        } else {
                            if (col < input.length()) {
                                lowerLayerValues[row] = input.get(col);
                            } else {
                                lowerLayerValues[row] = 1.0;
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
                        throw new RuntimeException(String.format("Problem in (varIndex/totalVars, layerIndex/totalLayers, row, col) = (%d/%d, %d/%d, %d, %d)",
                                                                 varIndex, variables.length, layerIndex, layers.length, row, col), e);
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
                    final ParameterMatrix layerWeights = layers[l].weights();
                    if (layerWeights.containsVariable(variable)) {
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
