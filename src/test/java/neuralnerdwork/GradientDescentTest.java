package neuralnerdwork;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.ShrinkingMode;
import net.jqwik.api.constraints.Size;
import neuralnerdwork.backprop.FeedForwardNetwork;
import neuralnerdwork.backprop.FullyConnectedLayer;
import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.MatrixFeatures_DDRM;

import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GradientDescentTest {

    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_rows = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_cols = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_layers = 10;
    private static final int singleDescentStepWithGenericDerivativeShouldReduceError_parameters =
            singleDescentStepWithGenericDerivativeShouldReduceError_rows
                    * (singleDescentStepWithGenericDerivativeShouldReduceError_cols + 1)
                    * singleDescentStepWithGenericDerivativeShouldReduceError_layers;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void singleDescentStepWithGenericDerivativeShouldReduceError(@ForAll @Size(value = singleDescentStepWithGenericDerivativeShouldReduceError_parameters) @Weight double[] values,
                                                                 @ForAll @Size(value = singleDescentStepWithGenericDerivativeShouldReduceError_cols) @TrainingInput double[] inputs,
                                                                 @ForAll @TrainingOutput @Size(singleDescentStepWithGenericDerivativeShouldReduceError_rows) double[] outputValues) {
        final int rows = singleDescentStepWithGenericDerivativeShouldReduceError_rows;
        final int cols = singleDescentStepWithGenericDerivativeShouldReduceError_cols;
        final int numLayers = singleDescentStepWithGenericDerivativeShouldReduceError_layers;

        final Model builder = new Model();
        final DMatrix trainingInput = new DMatrixRMaj(inputs);

        final LogisticFunction logistic = new LogisticFunction();

        final DMatrixRMaj biasComponent = new DMatrixRMaj(new double[]{1.0});
        VectorExpression networkFunction = new RowVectorConcat(new DMatrixColumnVectorExpression(trainingInput), new DMatrixColumnVectorExpression(biasComponent));
        for (int i = 0; i < numLayers-1; i++) {
            networkFunction =
                    new RowVectorConcat(
                            new ColumnVectorizedSingleVariableFunction(
                                    logistic,
                                    new MatrixVectorProduct(
                                            builder.createParameterMatrix(rows, cols + 1),
                                            networkFunction
                                    )
                            ),
                            new DMatrixColumnVectorExpression(biasComponent)
                    );
        }
        networkFunction =
                new ColumnVectorizedSingleVariableFunction(
                        logistic,
                        new MatrixVectorProduct(
                                builder.createParameterMatrix(rows, cols + 1),
                                networkFunction
                        )
                );

        final DMatrixRMaj trainingOutput = new DMatrixRMaj(outputValues);
        final VectorExpression error = VectorSum.sum(networkFunction, new ScaledVector(-1.0, new DMatrixColumnVectorExpression(trainingOutput)));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new DMatrixColumnVectorExpression(new DMatrixRMaj(ones)),
                new ColumnVectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final DMatrix lossDerivative = Util
                .logTiming("Computed derivative function", () -> squaredError.computeDerivative(parameterBindings
                ));

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }
        System.out.println();
        final double originalError = Util.logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(parameterBindings));
        Util.logFunctionStructure(lossDerivative);
        final double learningRate = 0.01;
        for (int i = 0; i < lossDerivative.getNumRows() * lossDerivative.getNumCols(); i++) {
            parameterBindings.put(i, parameterBindings.get(i) - learningRate * lossDerivative
                    .get(0, i));
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
            * singleDescentStepWithBackpropDerivativeShouldReduceError_layers;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void singleDescentStepWithBackpropDerivativeShouldReduceError(@ForAll @Size(value = singleDescentStepWithBackpropDerivativeShouldReduceError_parameters) @Weight double[] values,
                                                                 @ForAll @Size(value = singleDescentStepWithBackpropDerivativeShouldReduceError_cols) @TrainingInput double[] inputs,
                                                                 @ForAll @TrainingOutput @Size(singleDescentStepWithBackpropDerivativeShouldReduceError_rows) double[] outputValues) {
        final int rows = singleDescentStepWithBackpropDerivativeShouldReduceError_rows;
        final int cols = singleDescentStepWithBackpropDerivativeShouldReduceError_cols;
        final int numLayers = singleDescentStepWithBackpropDerivativeShouldReduceError_layers;

        final Model builder = new Model();
        final DMatrix trainingInput = new DMatrixRMaj(inputs);

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[singleDescentStepWithBackpropDerivativeShouldReduceError_layers];
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
        }

        final FeedForwardNetwork.FeedForwardExpression networkFunction = new FeedForwardNetwork(layers).expression(trainingInput);

        final DMatrix trainingOutput = new DMatrixRMaj(outputValues);
        final VectorExpression error = VectorSum.sum(networkFunction, new ScaledVector(-1.0, new DMatrixColumnVectorExpression(trainingOutput)));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new DMatrixColumnVectorExpression(new DMatrixRMaj(ones)),
                                                             new ColumnVectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        final DMatrix lossDerivative = Util
                .logTiming("Computed derivative function", () -> squaredError.computeDerivative(parameterBindings
                ));

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }
        System.out.println();
        final double originalError = Util.logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(parameterBindings));
        Util.logFunctionStructure(lossDerivative);
        final double learningRate = 0.01;
        for (int i = 0; i < lossDerivative.getNumRows() * lossDerivative.getNumCols(); i++) {
            parameterBindings.put(i, parameterBindings.get(i) - learningRate * lossDerivative
                    .get(0, i));
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
        final DMatrix trainingInput = new DMatrixRMaj(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[backpropShouldBeSameAsRegularGradient_layers];
        VectorExpression genericNetworkBuilder = new DMatrixColumnVectorExpression(trainingInput);
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
            genericNetworkBuilder =
                    new ColumnVectorizedSingleVariableFunction(
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
        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        final DMatrix genericResult = Util.logTiming("Evaluated generic derivative", () -> genericNetwork
                .computeDerivative(parameterBindings));
        final DMatrix specializedResult = Util
                .logTiming("Evaluated specialized derivative", () -> specializedImplementation
                        .computeDerivative(parameterBindings));

        final double delta = 1e-8;

        assertEquals(genericResult.getNumRows(), specializedResult.getNumRows());
        assertEquals(genericResult.getNumCols(), specializedResult.getNumCols());

        double[][] difference = new double[genericResult.getNumRows()][genericResult.getNumCols()];
        for (int row = 0; row < genericResult.getNumRows(); row++) {
            for (int col = 0; col < genericResult.getNumCols(); col++) {
                difference[row][col] = genericResult.get(row, col) - specializedResult.get(row, col);
            }
        }
        for (int row = 0; row < genericResult.getNumRows(); row++) {
            for (int col = 0; col < genericResult.getNumCols(); col++) {
                assertEquals(0.0, difference[row][col], delta,
                             () -> {
                                 var sb = new StringBuilder("Differences:\n");

                                 for (int r = 0; r < genericResult.getNumRows(); r++) {
                                     sb.append("[");
                                     for (int c = 0; c < genericResult.getNumCols(); c++) {
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
        final DMatrix trainingInput = new DMatrixRMaj(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FullyConnectedLayer[] layers = new FullyConnectedLayer[backpropSquaredErrorShoudlBeSameAsRegularGradientSquaredError_layers];
        VectorExpression genericNetworkBuilder = new DMatrixColumnVectorExpression(trainingInput);
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            ParameterVector bias = i < numLayers - 1 ? builder.createParameterVector(rows) : null;
            layers[i] = new FullyConnectedLayer(weights, Optional.ofNullable(bias), logistic);
            genericNetworkBuilder =
                    new ColumnVectorizedSingleVariableFunction(
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
        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        TrainingSample sample = new TrainingSample(inputs, outputs);
        ScalarExpression genericError = squaredError(sample, genericNetwork);
        ScalarExpression backpropError = squaredError(sample, specializedImplementation);
        final DMatrix genericResult = Util.logTiming("Evaluated generic derivative", () -> genericError
                .computeDerivative(parameterBindings));
        final DMatrix specializedResult = Util
                .logTiming("Evaluated specialized derivative", () -> backpropError
                        .computeDerivative(parameterBindings));

        final double delta = 1e-8;

        assertTrue(MatrixFeatures_DDRM.isEquals((DMatrixRMaj) genericResult, (DMatrixRMaj) specializedResult, delta), () ->
                "expected:\n" + genericResult + "\n\nobserved:\n" + specializedResult + "\n"
        );
    }

    private static ScalarExpression squaredError(TrainingSample sample, VectorExpression network) {
        // difference between network output and expected output
        final VectorExpression inputError = VectorSum.sum(network, new ScaledVector(-1.0, new DMatrixColumnVectorExpression(new DMatrixRMaj(sample.output()))));

        final double[] ones = new double[sample.output().length];
        Arrays.fill(ones, 1.0);
        return new DotProduct(new DMatrixColumnVectorExpression(new DMatrixRMaj(ones)),
                              new ColumnVectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), inputError));
    }
}
