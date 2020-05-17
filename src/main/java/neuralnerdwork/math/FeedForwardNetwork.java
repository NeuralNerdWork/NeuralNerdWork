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
            public Matrix evaluate(Model.Binder bindings) {
                record LayerEval(double[] activations, double[] activationInputs) {}
                final LayerEval[] feedForwardEvaluations = new LayerEval[layers.length];
                final ConstantVector biasComponent = new ConstantVector(new double[]{1.0});
                double[] lastOutput = input.toArray();
                // Feed forward
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
                    try {
                        final MatrixExpression paddedWeights =
                                new ZeroPaddedMatrix(
                                        layers[l + 1].weights(),
                                        layers.length - l - 2,
                                        layers.length - l - 2
                                );
                        final MatrixExpression paddedDiagonalActivationDerivative =
                                new ZeroPaddedMatrix(
                                        new DiagonalizedVector(
                                                new VectorizedSingleVariableFunction(
                                                        curActivationDerivative,
                                                        new ConstantVector(feedForwardEvaluations[l].activationInputs())
                                                )
                                        ),
                                        layers.length - l - 1,
                                        layers.length - l - 1
                                );
                        deltas[l] =
                                new MatrixProduct(
                                        new MatrixProduct(
                                                new ConstantMatrixExpression(deltas[l + 1]),
                                                paddedWeights
                                        ),
                                        paddedDiagonalActivationDerivative
                                ).evaluate(bindings);
                    } catch (IllegalArgumentException iae) {
                        throw new RuntimeException("Problem building deltas in layer " + l, iae);
                    }
                }

                // Build up derivatives using deltas and activations
                final double[][] partialDerivatives = new double[variables.length][];
                for (int varIndex = variables.length - 1; varIndex > -1; varIndex--) {
//                for (int varIndex = 0; varIndex < variables.length; varIndex++) {
                    final int variable = variables[varIndex];
                    final int layerIndex = findLayerIndex(variable);
                    final int row = layers[layerIndex].weights().rowIndexFor(variable);
                    final int col = layers[layerIndex].weights().colIndexFor(variable);

                    try {
                        final double[] lowerLayerValues = (layerIndex == layers.length - 1) ?
                                                          new double[layers[layerIndex].weights().rows()] :
                                                          new double[layers[layerIndex].weights().rows() + 1];
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
                                        new ZeroPaddedVector(
                                                new ConstantVector(lowerLayerValues),
                                                layerIndex == layers.length - 1 ?
                                                0 :
                                                (layers.length - 1) - layerIndex - 1
                                        )
                                ).evaluate(bindings)
                                 .toArray();
                    } catch (RuntimeException e) {
                        throw new RuntimeException(String.format("Problem in (varIndex/totalVars, layerIndex/totalLayers, row, col) = (%d/%d, %d/%d, %d, %d)",
                                                                 varIndex, variables.length, layerIndex, layers.length, row, col), e);
                    }
                }

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
