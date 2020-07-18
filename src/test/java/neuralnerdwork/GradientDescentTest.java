package neuralnerdwork;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.ShrinkingMode;
import net.jqwik.api.constraints.Size;
import neuralnerdwork.backprop.FeedForwardNetwork;
import neuralnerdwork.backprop.FullyConnectedLayer;
import neuralnerdwork.math.*;

import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class GradientDescentTest {

    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_rows = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_cols = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_layers = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_parameters =
            singleDescentStepWithGenericDerivativeShouldReduceError_rows
                    * (singleDescentStepWithGenericDerivativeShouldReduceError_cols + 1)
                    * singleDescentStepWithGenericDerivativeShouldReduceError_layers
                    + singleDescentStepWithGenericDerivativeShouldReduceError_cols;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void singleDescentStepWithGenericDerivativeShouldReduceError(@ForAll @Size(value = singleDescentStepWithGenericDerivativeShouldReduceError_parameters) @Weight double[] values,
                           @ForAll @TrainingOutput @Size(singleDescentStepWithGenericDerivativeShouldReduceError_rows) double[] outputValues) {
        final int rows = singleDescentStepWithGenericDerivativeShouldReduceError_rows;
        final int cols = singleDescentStepWithGenericDerivativeShouldReduceError_cols;
        final int numLayers = singleDescentStepWithGenericDerivativeShouldReduceError_layers;

        final Model builder = new Model();
        final VectorExpression trainingInput = builder.createParameterVector(cols);

        final LogisticFunction logistic = new LogisticFunction();

        final ConstantVector biasComponent = new ConstantVector(new double[]{1.0});
        VectorExpression networkFunction = new VectorConcat(trainingInput, biasComponent);
        for (int i = 0; i < numLayers-1; i++) {
            networkFunction =
                    new VectorConcat(
                            new VectorizedSingleVariableFunction(
                                    logistic,
                                    new MatrixVectorProduct(
                                            builder.createParameterMatrix(rows, cols + 1),
                                            networkFunction
                                    )
                            ),
                            biasComponent
                    );
        }
        networkFunction =
                new VectorizedSingleVariableFunction(
                        logistic,
                        new MatrixVectorProduct(
                                builder.createParameterMatrix(rows, cols + 1),
                                networkFunction
                        )
                );

        final ConstantVector trainingOutput = new ConstantVector(outputValues);
        final VectorExpression error = VectorSum.sum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final VectorExpression lossDerivative = Util
                .logTiming("Computed derivative function", () -> squaredError.computeDerivative(parameterBindings,
                                                                                                Arrays.copyOfRange(builder.variables(), trainingInput
                                                                                                        .length(), builder.size())
                ));

        final int[] vars = parameterBindings.variables();
        for (int i = 0; i < vars.length; i++) {
            parameterBindings.put(vars[i], values[i]);
        }
        System.out.println();
        final double originalError = Util.logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(parameterBindings));
        Util.logFunctionStructure(lossDerivative);
        final Vector updatedWeights = Util.logTiming("Applied derivative function", () -> lossDerivative.evaluate(parameterBindings));
        final double learningRate = 0.01;
        for (int i = 0; i < updatedWeights.length(); i++) {
            final int weightVariableIndex = trainingInput.length() + i;
            parameterBindings.put(weightVariableIndex, parameterBindings.get(weightVariableIndex) - learningRate * updatedWeights.get(i));
        }
        final double updated = squaredError.evaluate(parameterBindings);
        assertTrue(updated <= originalError, String.format("Expected error to decrease, but observed (original, updated) = (%f, %f)", originalError, updated));
        System.out.println();
    }

    private static final int singleDescentStepWithBackpropDerivativeShouldReduceError_rows = 10;
    private static final int singleDescentStepWithBackpropDerivativeShouldReduceError_cols = 10;
    private static final int singleDescentStepWithBackpropDerivativeShouldReduceError_layers = 50;
    private static final int singleDescentStepWithBackpropDerivativeShouldReduceError_parameters =
            singleDescentStepWithBackpropDerivativeShouldReduceError_rows
            * (singleDescentStepWithBackpropDerivativeShouldReduceError_cols + 1)
            * singleDescentStepWithBackpropDerivativeShouldReduceError_layers
            + singleDescentStepWithBackpropDerivativeShouldReduceError_cols;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void singleDescentStepWithBackpropDerivativeShouldReduceError(@ForAll @Size(value = singleDescentStepWithBackpropDerivativeShouldReduceError_parameters) @Weight double[] values,
                                                                 @ForAll @TrainingOutput @Size(singleDescentStepWithBackpropDerivativeShouldReduceError_rows) double[] outputValues) {
        final int rows = singleDescentStepWithBackpropDerivativeShouldReduceError_rows;
        final int cols = singleDescentStepWithBackpropDerivativeShouldReduceError_cols;
        final int numLayers = singleDescentStepWithBackpropDerivativeShouldReduceError_layers;

        final Model builder = new Model();
        final ConstantVector trainingInput = new ConstantVector(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[singleDescentStepWithBackpropDerivativeShouldReduceError_layers];
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
        }

        final FeedForwardNetwork.FeedForwardExpression networkFunction = new FeedForwardNetwork(layers).expression(trainingInput);

        final ConstantVector trainingOutput = new ConstantVector(outputValues);
        final VectorExpression error = VectorSum.sum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                                                             new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final VectorExpression lossDerivative = Util
                .logTiming("Computed derivative function", () -> squaredError.computeDerivative(parameterBindings,
                                                                                                Arrays.copyOfRange(builder.variables(), trainingInput
                                                                                                        .length(), builder.size())
                ));

        final int[] vars = parameterBindings.variables();
        for (int i = 0; i < vars.length; i++) {
            parameterBindings.put(vars[i], values[i]);
        }
        System.out.println();
        final double originalError = Util.logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(parameterBindings));
        Util.logFunctionStructure(lossDerivative);
        final Vector updatedWeights = Util.logTiming("Applied derivative function", () -> lossDerivative.evaluate(parameterBindings));
        final double learningRate = 0.01;
        for (int i = 0; i < updatedWeights.length(); i++) {
            final int weightVariableIndex = trainingInput.length() + i;
            parameterBindings.put(weightVariableIndex, parameterBindings.get(weightVariableIndex) - learningRate * updatedWeights.get(i));
        }
        final double updated = squaredError.evaluate(parameterBindings);
        assertTrue(updated <= originalError, String.format("Expected error to decrease, but observed (original, updated) = (%f, %f)", originalError, updated));
        System.out.println();
    }

    private static final int backpropShouldBeSameAsRegularGradient_rows = 2;
    private static final int backpropShouldBeSameAsRegularGradient_cols = 2;
    private static final int backpropShouldBeSameAsRegularGradient_layers = 10;
    private static final int backpropShouldBeSameAsRegularGradient_parameters =
            backpropShouldBeSameAsRegularGradient_rows
            * (backpropShouldBeSameAsRegularGradient_cols + 1)
            * backpropShouldBeSameAsRegularGradient_layers
            + backpropShouldBeSameAsRegularGradient_cols;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void backpropShouldBeSameAsRegularGradient(@ForAll @Size(value = backpropShouldBeSameAsRegularGradient_parameters) @Weight double[] values) {
        final int rows = backpropShouldBeSameAsRegularGradient_rows;
        final int cols = backpropShouldBeSameAsRegularGradient_cols;
        final int numLayers = backpropShouldBeSameAsRegularGradient_layers;

        final Model builder = new Model();
        final ConstantVector trainingInput = new ConstantVector(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[backpropShouldBeSameAsRegularGradient_layers];
        VectorExpression genericNetworkBuilder = trainingInput;
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
            genericNetworkBuilder =
                    new VectorizedSingleVariableFunction(
                            logistic,
                            bias != null ?
                                    VectorSum.sum(
                                            MatrixVectorProduct.product(
                                                    weights,
                                                    genericNetworkBuilder
                                            ),
                                            bias
                                    ) :
                                    MatrixVectorProduct.product(
                                            weights,
                                            genericNetworkBuilder
                                    )
                    );
        }

        final VectorExpression genericNetwork = genericNetworkBuilder;
        final FeedForwardNetwork.FeedForwardExpression specializedImplementation = new FeedForwardNetwork(layers).expression(trainingInput);

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final int[] vars = parameterBindings.variables();
        for (int i = 0; i < vars.length; i++) {
            parameterBindings.put(vars[i], values[i]);
        }

        final Matrix genericResult = Util.logTiming("Evaluated generic derivative", () -> genericNetwork
                .computeDerivative(parameterBindings, parameterBindings.variables()));
        final Matrix specializedResult = Util
                .logTiming("Evaluated specialized derivative", () -> specializedImplementation
                        .computeDerivative(parameterBindings, parameterBindings.variables()));

        final double delta = 1e-8;

        assertEquals(genericResult.rows(), specializedResult.rows());
        assertEquals(genericResult.cols(), specializedResult.cols());

        double[][] difference = new double[genericResult.rows()][genericResult.cols()];
        for (int row = 0; row < genericResult.rows(); row++) {
            for (int col = 0; col < genericResult.cols(); col++) {
                difference[row][col] = genericResult.get(row, col) - specializedResult.get(row, col);
            }
        }
        for (int row = 0; row < genericResult.rows(); row++) {
            for (int col = 0; col < genericResult.cols(); col++) {
                assertEquals(0.0, difference[row][col], delta,
                             () -> {
                                 var sb = new StringBuilder("Differences:\n");

                                 for (int r = 0; r < genericResult.rows(); r++) {
                                     sb.append("[");
                                     for (int c = 0; c < genericResult.cols(); c++) {
                                         double rawDiff = difference[r][c];
                                         sb.append(rawDiff);
                                         sb.append(", ");
                                     }
                                     sb.delete(sb.length() - 2, sb.length());
                                     sb.append("]\n");
                                 }

                                 return sb.toString();
                             });
            }
        }
    }

    private static final int backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_rows = 2;
    private static final int backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols = 2;
    private static final int backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_layers = 10;
    private static final int backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_parameters =
            backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_rows
                    * (backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols + 1)
                    * backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_layers
                    + backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError(@ForAll @Size(value = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_parameters) @Weight double[] values,
                                                                       @ForAll @Size(value = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols) @TrainingInput double[] inputs,
                                                                       @ForAll @Size(value = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols) @TrainingOutput double[] outputs) {
        final int rows = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_rows;
        final int cols = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_cols;
        final int numLayers = backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_layers;

        final Model builder = new Model();
        final ConstantVector trainingInput = new ConstantVector(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_layers];
        VectorExpression genericNetworkBuilder = trainingInput;
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
            genericNetworkBuilder =
                    new VectorizedSingleVariableFunction(
                            logistic,
                            bias != null ?
                                    VectorSum.sum(
                                            MatrixVectorProduct.product(
                                                    weights,
                                                    genericNetworkBuilder
                                            ),
                                            bias
                                    ) :
                                    MatrixVectorProduct.product(
                                            weights,
                                            genericNetworkBuilder
                                    )
                    );
        }

        final VectorExpression genericNetwork = genericNetworkBuilder;
        final FeedForwardNetwork.FeedForwardExpression specializedImplementation = new FeedForwardNetwork(layers).expression(trainingInput);

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final int[] vars = parameterBindings.variables();
        for (int i = 0; i < vars.length; i++) {
            parameterBindings.put(vars[i], values[i]);
        }

        TrainingSample sample = new TrainingSample(new ConstantVector(inputs), new ConstantVector(outputs));
        ScalarExpression genericError = squaredError(sample, genericNetwork);
        ScalarExpression backpropError = squaredError(sample, specializedImplementation);
        final Vector genericResult = Util.logTiming("Evaluated generic derivative", () -> genericError
                .computeDerivative(parameterBindings, parameterBindings.variables()));
        final Vector specializedResult = Util
                .logTiming("Evaluated specialized derivative", () -> backpropError
                        .computeDerivative(parameterBindings, parameterBindings.variables()));

        final double delta = 1e-8;

        assertEquals(genericResult.length(), specializedResult.length());
        assertArrayEquals(genericResult.toArray(), specializedResult.toArray(), delta);
    }

    private static ScalarExpression squaredError(TrainingSample sample, VectorExpression network) {
        // difference between network output and expected output
        final VectorExpression inputError = VectorSum.sum(network, new ScaledVector(-1.0, sample.output()));

        final double[] ones = new double[sample.output().length()];
        Arrays.fill(ones, 1.0);
        return new DotProduct(new ConstantVector(ones),
                              new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), inputError));
    }
}
