package neuralnerdwork;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.ShrinkingMode;
import net.jqwik.api.constraints.Size;
import neuralnerdwork.math.*;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GradientDescentTest {

    private static final int singleDescentStep_rows = 10;
    private static final int singleDescentStep_cols = 10;
    private static final int singleDescentStep_layers = 50;
    private static final int singleDescentStep_parameters =
            singleDescentStep_rows
                    * (singleDescentStep_cols + 1)
                    * singleDescentStep_layers
                    + singleDescentStep_cols;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void singleDescentStep(@ForAll @Size(value = singleDescentStep_parameters) @Weight double[] values,
                           @ForAll @TrainingOutput @Size(singleDescentStep_rows) double[] outputValues) {
        final int rows = singleDescentStep_rows;
        final int cols = singleDescentStep_cols;
        final int numLayers = singleDescentStep_layers;

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
        final VectorSum error = new VectorSum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final VectorExpression lossDerivative = Util.logTiming("Computed derivative function", () -> squaredError.computeDerivative(
                Arrays.copyOfRange(builder.variables(), trainingInput.length(), builder.size())
        ));

        final Model.Binder binder = builder.createBinder();
        final int[] vars = binder.variables();
        for (int i = 0; i < vars.length; i++) {
            binder.put(vars[i], values[i]);
        }
        System.out.println();
        final double originalError = Util.logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(binder));
        Util.logFunctionStructure(lossDerivative);
        final Vector updatedWeights = Util.logTiming("Applied derivative function", () -> lossDerivative.evaluate(binder));
        final double learningRate = 0.01;
        for (int i = 0; i < updatedWeights.length(); i++) {
            final int weightVariableIndex = trainingInput.length() + i;
            binder.put(weightVariableIndex, binder.get(weightVariableIndex) - learningRate * updatedWeights.get(i));
        }
        final double updated = squaredError.evaluate(binder);
        assertTrue(updated <= originalError, String.format("Expected error to decrease, but observed (original, updated) = (%f, %f)", originalError, updated));
        System.out.println();
    }

    private static final int backpropShouldBeSameAsRegularGradient_rows = 2;
    private static final int backpropShouldBeSameAsRegularGradient_cols = 2;
    private static final int backpropShouldBeSameAsRegularGradient_layers = 3;
    private static final int backpropShouldBeSameAsRegularGradient_parameters =
            backpropShouldBeSameAsRegularGradient_rows
            * (backpropShouldBeSameAsRegularGradient_cols + 1)
            * backpropShouldBeSameAsRegularGradient_layers;
    @Property(tries = 3, shrinking = ShrinkingMode.OFF)
    void backpropShouldBeSameAsRegularGradient(@ForAll @Size(value = backpropShouldBeSameAsRegularGradient_parameters) @Weight double[] values,
                           @ForAll @TrainingOutput @Size(backpropShouldBeSameAsRegularGradient_rows) double[] outputValues) {
        final int rows = backpropShouldBeSameAsRegularGradient_rows;
        final int cols = backpropShouldBeSameAsRegularGradient_cols;
        final int numLayers = backpropShouldBeSameAsRegularGradient_layers;

        final Model builder = new Model();
        final ConstantVector trainingInput = new ConstantVector(Arrays.copyOf(values, cols));

        final LogisticFunction logistic = new LogisticFunction();

        final FeedForwardNetwork.Layer[] layers = new FeedForwardNetwork.Layer[backpropShouldBeSameAsRegularGradient_layers];
        VectorExpression genericNetworkBuilder = trainingInput;
        for (int i = 0; i < numLayers; i++) {
            ParameterMatrix weights = builder.createParameterMatrix(rows, cols);
            layers[i] = new FeedForwardNetwork.Layer(weights, logistic);
            genericNetworkBuilder =
                    new VectorizedSingleVariableFunction(
                            logistic,
                            new MatrixVectorProduct(
                                    weights,
                                    genericNetworkBuilder
                            )
                    );
        }

        final VectorExpression genericNetwork = genericNetworkBuilder;
        final FeedForwardNetwork specializedImplementation = new FeedForwardNetwork(trainingInput, layers);

        final Model.Binder binder = builder.createBinder();
        final int[] vars = binder.variables();
        for (int i = 0; i < vars.length; i++) {
            binder.put(vars[i], values[i]);
        }

        final MatrixExpression genericDerivative = genericNetwork.computeDerivative(binder.variables());
        final MatrixExpression specializedDerivative = specializedImplementation.computeDerivative(binder.variables());

        final Matrix genericResult = Util.logTiming("Evaluated generic derivative", () -> genericDerivative.evaluate(binder));
        final Matrix specializedResult = Util.logTiming("Evaluated specialized derivative", () -> specializedDerivative.evaluate(binder));

        final double delta = 0.0000001;

        assertEquals(genericResult.rows(), specializedResult.rows());
        assertEquals(genericResult.cols(), specializedResult.cols());

        for (int row = 0; row < genericResult.rows(); row++) {
            for (int col = 0; col < genericResult.cols(); col++) {
                assertEquals(genericResult.get(row, col), specializedResult.get(row, col), delta,
                             () -> String.format("Generic result:\n%s\n\nSpecialized result:\n%s\n",
                                                 Arrays.deepToString(genericResult.toArray()),
                                                 Arrays.deepToString(specializedResult.toArray())));
            }
        }
    }
}
