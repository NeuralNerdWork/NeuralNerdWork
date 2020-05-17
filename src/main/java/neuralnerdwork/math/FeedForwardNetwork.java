package neuralnerdwork.math;

public record FeedForwardNetwork(ConstantVector input, Layer[] layers) implements VectorExpression {
    public record Layer(ParameterMatrix weights, SingleVariableFunction activation) {}
    @Override
    public int length() {
        return layers[layers.length-1].weights().rows();
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        VectorExpression network = input;
        for (int l = 0; l < layers.length; l++) {
            final Layer layer = layers[l];
            network = new VectorizedSingleVariableFunction(
                    layer.activation(),
                    new MatrixVectorProduct(
                            layer.weights(),
                            network
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
            public Matrix evaluate(Model.Binder bindings) {
                record LayerEval(double[] activations, double[] activationInputs) {}
                final LayerEval[] feedForwardEvaluations = new LayerEval[layers.length];
                double[] lastOutput = input.toArray();
                // Feed forward
                for (int l = 0; l < layers.length; l++) {
                    final Layer layer = layers[l];
                    final double[] curActivationInputs = new MatrixVectorProduct(
                            layer.weights(),
                            new ConstantVector(lastOutput)
                    ).evaluate(bindings)
                     .toArray();

                    lastOutput = new VectorizedSingleVariableFunction(
                            layer.activation(),
                            new ConstantVector(curActivationInputs)
                    ).evaluate(bindings)
                     .toArray();

                    feedForwardEvaluations[l] = new LayerEval(lastOutput, curActivationInputs);
                }

                // Backprop
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
                    deltas[l] = new MatrixProduct(
                            new MatrixProduct(
                                    new ConstantMatrixExpression(deltas[l + 1]),
                                    layers[l + 1].weights()
                            ),
                            new DiagonalizedVector(
                                    new VectorizedSingleVariableFunction(
                                            curActivationDerivative,
                                            new ConstantVector(feedForwardEvaluations[l].activationInputs())
                                    )
                            )
                    ).evaluate(bindings);
                }

                final double[][] partialDerivatives = new double[variables.length][];
                for (int varIndex = 0; varIndex < variables.length; varIndex++) {
                    final int variable = variables[varIndex];
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

                    final int row = layers[l].weights().rowIndexFor(variable);
                    final int col = layers[l].weights().colIndexFor(variable);

                    final double[] lowerLayerValues = new double[layers[l].weights().cols()];
                    if (l > 0) {
                        lowerLayerValues[row] = feedForwardEvaluations[l-1].activations()[col];
                    } else {
                        lowerLayerValues[row] = input.get(col);

                    }
                    final ConstantMatrixExpression layerDeltas = new ConstantMatrixExpression(deltas[l]);
                    partialDerivatives[varIndex] =
                            new MatrixVectorProduct(
                                    layerDeltas,
                                    new ConstantVector(lowerLayerValues)
                            ).evaluate(bindings)
                             .toArray();
                }

                return new Transpose(new ConstantArrayMatrix(partialDerivatives));
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
