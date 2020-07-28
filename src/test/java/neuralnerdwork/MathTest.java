package neuralnerdwork;

import net.jqwik.api.Example;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.ShrinkingMode;
import net.jqwik.api.constraints.Size;
import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.String.format;
import static neuralnerdwork.math.VectorSum.sum;
import static org.junit.jupiter.api.Assertions.*;

public class MathTest {

    @Property(shrinking = ShrinkingMode.OFF)
    void twoMatrixPlusVectorDerivativeInOuterMatrix(@ForAll @Size(value = 2 + 2*2 + 2*2) double[] values) {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7Bi%7D%2C%7Bj%7D%7D+by+b
        final Model builder = new Model();
        final VectorExpression vector = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final ParameterMatrix w2 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct multiplication = new MatrixVectorProduct(
                new MatrixProduct(
                        w2,
                        w1
                ),
                vector
        );

        final int variable = w2.variableIndexFor(0, 1);
        final Model.ParameterBindings parameterBindings = builder.createBinder();

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        final Vector inputEval = vector.evaluate(parameterBindings);
        final Vector derivativeVector = multiplication.computePartialDerivative(parameterBindings, variable);
        final double firstVar = parameterBindings.get(w1.variableIndexFor(1, 0));
        final double secondVar = parameterBindings.get(w1.variableIndexFor(1, 1));

        assertEquals(2, derivativeVector.length(), "Length not equal");
        assertEquals(inputEval.get(0) * firstVar + inputEval.get(1) * secondVar, derivativeVector.get(0), 0.0001);
        assertEquals(0.0, derivativeVector.get(1), 0.0001);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void twoMatrixPlusVectorDerivativeInInnerMatrix(@ForAll @Size(value = 2 + 2*2 + 2*2) double[] values) {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+f
        final Model builder = new Model();
        final VectorExpression vector = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final ParameterMatrix w2 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct multiplication = new MatrixVectorProduct(
                new MatrixProduct(
                        w2,
                        w1
                ),
                vector
        );

        final int variable = w1.variableIndexFor(0, 1);
        final Model.ParameterBindings parameterBindings = builder.createBinder();

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        final Vector inputEval = vector.evaluate(parameterBindings);
        final Vector derivativeVector = multiplication.computePartialDerivative(parameterBindings, variable);
        final double firstVar = parameterBindings.get(w2.variableIndexFor(0, 0));
        final double secondVar = parameterBindings.get(w2.variableIndexFor(1, 0));

        assertEquals(2, derivativeVector.length(), "Length not equal");
        assertEquals(inputEval.get(1) * firstVar, derivativeVector.get(0), 0.0001);
        assertEquals(inputEval.get(1) * secondVar, derivativeVector.get(1), 0.0001);
    }

    /*
     * Same as #twoMatrixPlusVectorDerivativeInInnerMatrix but with different multiplication object tree
     * to test associativity
     */
    @Property(shrinking = ShrinkingMode.OFF)
    void associativityOfDerivative(@ForAll @Size(value = 2*2 + 2*2) @Weight double[] weights, @ForAll @Size(2) @TrainingInput double[] inputs) {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7Bi%7D%2C%7Bj%7D%7D+by+f
        final Model builder = new Model();
        final VectorExpression vector = new ConstantVector(inputs);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final ParameterMatrix w2 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct multiplication = new MatrixVectorProduct(
                w2,
                new MatrixVectorProduct(
                        w1,
                        vector
                )
        );

        final int variable = w1.variableIndexFor(0, 1);
        final Model.ParameterBindings parameterBindings = builder.createBinder();

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, weights[i++]);
            }
        }
        final Vector inputEval = vector.evaluate(parameterBindings);
        final Vector derivativeVector = multiplication.computePartialDerivative(parameterBindings, variable);
        final double firstVar = parameterBindings.get(w2.variableIndexFor(0, 0));
        final double secondVar = parameterBindings.get(w2.variableIndexFor(1, 0));

        assertEquals(2, derivativeVector.length(), "Length not equal");
        assertEquals(inputEval.get(1) * firstVar, derivativeVector.get(0), 0.0001);
        assertEquals(inputEval.get(1) * secondVar, derivativeVector.get(1), 0.0001);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void partialDerviativeWeightsWithActivation(@ForAll @Size(value = 2 + 2*2) @Weight double[] values) {
        final Model builder = new Model();
        final VectorExpression vector = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                vector
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new LogisticFunction(),
                                                                                    weightedInputs);

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }
        final Vector observed = layerFunction.computePartialDerivative(parameterBindings, w1.variableIndexFor(0, 0));

        assertEquals(2, observed.length(), "Length not equal");
        final Vector inputEval = vector.evaluate(parameterBindings);
        final double logisticInput = parameterBindings.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + parameterBindings.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
        final double firstExpected = Util.logistic(logisticInput) * (1 - Util.logistic(logisticInput)) * inputEval.get(0);
        assertEquals(firstExpected, observed.get(0), 0.0001);
        assertEquals(0.0, observed.get(1), 0.0001);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void fullDerivativeWeightsWithActivation(@ForAll @Size(value = 2 + 2*2) @Weight double[] values) {
        final Model builder = new Model();
        final Vector vector = new ConstantVector(Arrays.copyOfRange(values, 4, 6));
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                vector
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new LogisticFunction(),
                                                                                    weightedInputs);

        final Model.ParameterBindings parameterBindings = builder.createBinder();

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }
        final DMatrix observed = layerFunction.computeDerivative(parameterBindings);

        assertEquals(2, observed.getNumRows(), "Rows not equal");
        assertEquals(4, observed.getNumCols(), "Cols not equal");

        final Vector inputEval = vector.evaluate(parameterBindings);
        final double firstLogisticInput = parameterBindings.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + parameterBindings.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
        final double secondLogisticInput = parameterBindings.get(w1.variableIndexFor(1, 0)) * inputEval.get(0) + parameterBindings.get(w1.variableIndexFor(1, 1)) * inputEval.get(1);

        final double firstExpected = Util.logistic(firstLogisticInput) * Util.logistic(-firstLogisticInput) * inputEval.get(0);
        final double secondExpected = Util.logistic(firstLogisticInput) * Util.logistic(-firstLogisticInput) * inputEval.get(1);
        final double thirdExpected = Util.logistic(secondLogisticInput) * Util.logistic(-secondLogisticInput) * inputEval.get(0);
        final double fourthExpected = Util.logistic(secondLogisticInput) * Util.logistic(-secondLogisticInput) * inputEval.get(1);

        assertEquals(firstExpected, observed.get(0, 0), 0.0001);
        assertEquals(0.0, observed.get(1, 0), 0.0001);

        assertEquals(secondExpected, observed.get(0, 1), 0.0001);
        assertEquals(0.0, observed.get(1, 1), 0.0001);

        assertEquals(0.0, observed.get(0, 2), 0.0001);
        assertEquals(thirdExpected, observed.get(1, 2), 0.0001);

        assertEquals(0.0, observed.get(0, 3), 0.0001);
        assertEquals(fourthExpected, observed.get(1, 3), 0.0001);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void fullDerivativeOfLossFunctionOnLayer(@ForAll @Size(value = 2*2) @Weight double[] values, @ForAll @Size(value = 2) @TrainingInput double[] inputs) {
        final Model builder = new Model();
        final Vector trainingInput = new ConstantVector(inputs);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                trainingInput
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new LogisticFunction(),
                                                                                    weightedInputs);

        final ConstantVector trainingOutput = new ConstantVector(new double[] { 1.0, 0.0 });
        final VectorExpression error = sum(layerFunction, new ScaledVector(-1.0, trainingOutput));
        // TODO should make a square combinator; this will differentiate error twice
        final DotProduct squaredError = new DotProduct(error, error);

        final Model.ParameterBindings parameterBindings = builder.createBinder();

        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }
        final Vector observed = squaredError.computeDerivative(parameterBindings);

        assertEquals(4, observed.length(), "Length not equal");

        final Vector inputEval = trainingInput.evaluate(parameterBindings);
        final double unactivatedNeuron1 = parameterBindings.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + parameterBindings.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
        final double unactivatedNeuron2 = parameterBindings.get(w1.variableIndexFor(1, 0)) * inputEval.get(0) + parameterBindings.get(w1.variableIndexFor(1, 1)) * inputEval.get(1);

        final double[] expected = new double[] {
                2.0 * (Util.logistic(unactivatedNeuron1) - trainingOutput.get(0)) * Util.logisticDerivative(unactivatedNeuron1) * inputEval.get(0),
                2.0 * (Util.logistic(unactivatedNeuron1) - trainingOutput.get(0)) * Util.logisticDerivative(unactivatedNeuron1) * inputEval.get(1),
                2.0 * (Util.logistic(unactivatedNeuron2) - trainingOutput.get(1)) * Util.logisticDerivative(unactivatedNeuron2) * inputEval.get(0),
                2.0 * (Util.logistic(unactivatedNeuron2) - trainingOutput.get(1)) * Util.logisticDerivative(unactivatedNeuron2) * inputEval.get(1)
        };

        assertArrayEquals(expected, observed.toArray(), 0.0001, format("Expected: %s, Observed: %s", Arrays.toString(expected), Arrays.toString(observed.toArray())));
    }

    @Property(tries = 3)
    void noZeroValuesInDerivative(@ForAll @Size(10*10*5) @Weight double[] weights, @ForAll @Size(value = 10) @TrainingOutput double[] outputValues) {
        final int rows = 10;
        final int cols = 10;
        final int numLayers = 5;

        final Model builder = new Model();
        final VectorExpression trainingInput = builder.createParameterVector(cols);

        final LogisticFunction logistic = new LogisticFunction();

        VectorExpression networkFunction = trainingInput;
        for (int i = 0; i < numLayers; i++) {
            networkFunction =
                    new VectorizedSingleVariableFunction(
                            logistic,
                            new MatrixVectorProduct(
                                    builder.createParameterMatrix(rows, cols),
                                    networkFunction
                            )
                    );
        }

        final ConstantVector trainingOutput = new ConstantVector(outputValues);
        final VectorExpression error = sum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                                                             new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        Model.ParameterBindings binder = builder.createBinder();
        for (int i = 0; i < weights.length; i++) {
            binder.put(i, weights[i]);
        }
        final VectorExpression lossDerivative = Util.logTiming("Computed derivative function", () -> squaredError
                .computeDerivative(binder));
        assertNonZero(lossDerivative);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void convolutionFilterMatrixShouldGiveConvolutionFilterResultWhenMultipled(@ForAll @Weight @Size(value = 2*2 + 3*3) double[] values) {
        final Model builder = new Model();
        final ParameterMatrix filter = builder.createParameterMatrix(2, 2);
        final ConvolutionFilterMatrix convolutionMatrix = new ConvolutionFilterMatrix(filter, 3, 3);
        final Vector inputVector = new ConstantVector(Arrays.copyOfRange(values, 4, values.length));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        Vector convolutionResult = new MatrixVectorProduct(convolutionMatrix, inputVector).evaluate(parameterBindings);

        double[] expected = {
                values[0] * values[4] + values[1] * values[5] + values[2] * values[7] + values[3] * values[8],
                values[0] * values[5] + values[1] * values[6] + values[2] * values[8] + values[3] * values[9],
                values[0] * values[7] + values[1] * values[8] + values[2] * values[10] + values[3] * values[11],
                values[0] * values[8] + values[1] * values[9] + values[2] * values[11] + values[3] * values[12],
        };

        assertArrayEquals(expected,
                          convolutionResult.toArray(),
                          0.0001,
                          () -> "Expected: " + Arrays.toString(expected) + "\nObserved: " + Arrays
                                  .toString(convolutionResult.toArray()) +
                                  "\nValues: " + Arrays.toString(values) +
                                  "\nConvolution Matrix: " + convolutionMatrix.evaluate(parameterBindings) + "\n");
    }

    @Example
    void convolutionFilterMatrixWithMuchBiggerImageThanFilterShouldGiveReasonableResult() {
        final Model builder = new Model();
        final ParameterMatrix filter = builder.createParameterMatrix(5, 5);
        final ConvolutionFilterMatrix convolutionMatrix = new ConvolutionFilterMatrix(filter, 20, 20);
        double[] inputOnes = new double[20 * 20];
        Arrays.fill(inputOnes, 1.0);
        final Vector inputVector = new ConstantVector(inputOnes);

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        for (int var : parameterBindings.variables()) {
            parameterBindings.put(var, 1.0);
        }

        Vector convolutionResult = new MatrixVectorProduct(convolutionMatrix, inputVector).evaluate(parameterBindings);

        double[] expected = new double[16*16];
        Arrays.fill(expected, 25.0);

        assertArrayEquals(expected,
                          convolutionResult.toArray(),
                          0.0001,
                          () -> "Expected: " + Arrays.toString(expected) + "\nObserved: " + Arrays
                                  .toString(convolutionResult.toArray()) +
                                  "\nConvolution Matrix: " + convolutionMatrix.evaluate(parameterBindings) + "\n");
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void convolutionFilterSizeOneGivesSameResultBack(@ForAll @Weight @Size(value = 3*3) double[] values) {
        final Model builder = new Model();
        final ParameterMatrix filter = builder.createParameterMatrix(1, 1);
        final ConvolutionFilterMatrix convolutionMatrix = new ConvolutionFilterMatrix(filter, 3, 3);
        final Vector inputVector = new ConstantVector(Arrays.copyOf(values, values.length));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        parameterBindings.put(0, 1.0);

        Vector convolutionResult = new MatrixVectorProduct(convolutionMatrix, inputVector).evaluate(parameterBindings);

        assertArrayEquals(values,
                          convolutionResult.toArray(),
                          0.0001,
                          () -> "Expected: " + Arrays.toString(values) + "\nObserved: " + Arrays
                                  .toString(convolutionResult.toArray()) +
                                  "\nValues: " + Arrays.toString(values) +
                                  "\nConvolution Matrix: " + convolutionMatrix.evaluate(parameterBindings) + "\n");
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void convolutionFilterSameSizeAsInputGivesSumOfInputComponents(@ForAll @Weight @Size(value = 3*3) double[] values) {
        final Model builder = new Model();
        final ParameterMatrix filter = builder.createParameterMatrix(3, 3);
        final ConvolutionFilterMatrix convolutionMatrix = new ConvolutionFilterMatrix(filter, 3, 3);
        final Vector inputVector = new ConstantVector(Arrays.copyOf(values, values.length));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        for (int var : parameterBindings.variables()) {
            parameterBindings.put(var, 1.0);
        }

        Vector convolutionResult = new MatrixVectorProduct(convolutionMatrix, inputVector).evaluate(parameterBindings);
        assertEquals(1, convolutionResult.length());
        double observed = convolutionResult.get(0);

        assertEquals(Arrays.stream(values).sum(), observed, 1e-8);
    }

    @Property(shrinking = ShrinkingMode.OFF)
    void convolutionFilterMatrixShouldHaveCorrectDerivative(@ForAll @Weight @Size(value = 2*2 + 3*3) double[] values) {
        final Model builder = new Model();
        final ParameterMatrix filter = builder.createParameterMatrix(2, 2);
        final ConvolutionFilterMatrix convolutionMatrix = new ConvolutionFilterMatrix(filter, 3, 3);
        final Vector inputVector = new ConstantVector(Arrays.copyOfRange(values, 4, values.length));

        final Model.ParameterBindings parameterBindings = builder.createBinder();
        {
            int i = 0;
            for (int var : parameterBindings.variables()) {
                parameterBindings.put(var, values[i++]);
            }
        }

        Vector derivativeResult = new MatrixVectorProduct(convolutionMatrix, inputVector)
                .computePartialDerivative(parameterBindings, filter.variableIndexFor(0, 0));

        double[] expected = {
                values[4],
                values[5],
                values[7],
                values[8]
        };

        assertArrayEquals(expected,
                          derivativeResult.toArray(),
                          0.0001,
                          () -> "Expected: " + Arrays.toString(expected) + "\nObserved: " + Arrays
                                  .toString(derivativeResult.toArray()) +
                                  "\nValues: " + Arrays.toString(values) +
                                  "\nConvolution Matrix: " + convolutionMatrix.evaluate(parameterBindings) + "\n");
    }

    @Example
    void maxPoolOnSimpleVectorShouldGiveCorrectResult() {
        double[] flatValues = new double[]{
                2.0, 0.4,   0.1, 1.0,
                1.0, 1.0,   0.5, 0.2,

                8.0, 0.1,   2.0, 3.0,
                0.8, 1.0,   1.0, 0.0
        };

        MaxPoolVector maxPool = new MaxPoolVector(new ConstantVector(flatValues), 4, 4, 2, 2);
        Vector result = maxPool.evaluate(null);

        assertArrayEquals(new double[] { 2.0, 1.0, 8.0, 3.0 }, result.toArray(), 1e-8);
    }

    @Example
    void maxPoolPartialDerivativeGivesCorrectResult() {
        Model model = new Model();
        ParameterVector paramVector = model.createParameterVector(16);
        double[] flatValues = new double[]{
                2.0, 0.4,   0.1, 1.0,
                1.0, 1.0,   0.5, 0.2,

                8.0, 0.1,   2.0, 3.0,
                0.8, 1.0,   1.0, 0.0
        };
        Model.ParameterBindings binder = model.createBinder();
        for (int variable : binder.variables()) {
            binder.put(variable, 0.01);
        }

        VectorExpression maxPoolInput = new MatrixVectorProduct(
                new DiagonalizedVector(new ConstantVector(flatValues)),
                paramVector
        );

        MaxPoolVector maxPool = new MaxPoolVector(maxPoolInput, 4, 4, 2, 2);
        Vector result = maxPool.computePartialDerivative(binder, paramVector.variableFor(0));

        assertArrayEquals(new double[] { 2.0, 0.0, 0.0, 0.0 }, result.toArray(), 1e-8);
    }

    @Example
    void maxPoolDerivativeGivesCorrectResult() {
        Model model = new Model();
        ParameterVector paramVector = model.createParameterVector(16);
        double[] flatValues = new double[]{
                2.0, 0.4,   0.1, 1.0,
                1.0, 1.0,   0.5, 0.2,

                8.0, 0.1,   2.0, 3.0,
                0.8, 1.0,   1.0, 0.0
        };
        Model.ParameterBindings binder = model.createBinder();
        for (int variable : binder.variables()) {
            binder.put(variable, 0.01);
        }

        VectorExpression maxPoolInput = new MatrixVectorProduct(
                new DiagonalizedVector(new ConstantVector(flatValues)),
                paramVector
        );

        MaxPoolVector maxPool = new MaxPoolVector(maxPoolInput, 4, 4, 2, 2);
        DMatrix result = maxPool.computeDerivative(binder);

        double[][] expectedArray = new double[][] {
                {2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0},
        };
        DMatrixRMaj expected = new DMatrixRMaj(expectedArray);
        for (int r = 0; r < expected.getNumRows(); r++) {
            for (int c = 0; c < expected.getNumCols(); c++) {
                assertEquals(expected.get(r, c), result.get(r, c), 1e-08, () -> "expected:\n" + expected + "\n\nobserved:\n" + result + "\n");
            }
        }
    }

    private void assertNonZero(Object expression) {
        assertNonZero(expression, new ArrayList<>(List.of(expression.getClass().getSimpleName() + " root")));
    }

    private void assertNonZero(Object expression, List<String> path) {
        Arrays.stream(expression.getClass().getDeclaredFields())
              .filter(f -> VectorExpression.class.isAssignableFrom(f.getType())
                      || MatrixExpression.class.isAssignableFrom(f.getType())
                      || ScalarExpression.class.isAssignableFrom(f.getType()))
              .forEach(field -> {
                  try {
                      field.setAccessible(true);
                      final Object fieldInstance = field.get(expression);
                      path.add(fieldInstance.getClass().getSimpleName() + " " + field.getName());
                      if (fieldInstance instanceof VectorExpression v) {
                          assertFalse(v.isZero(), "path=" + path + ", value=" + fieldInstance);
                      } else if (fieldInstance instanceof MatrixExpression m) {
                          assertFalse(m.isZero(), "path=" + path + ", value=" + fieldInstance);
                      } else if (fieldInstance instanceof ScalarExpression s) {
                          assertFalse(s.isZero(), "path=" + path + ", value=" + fieldInstance);
                      }

                      assertNonZero(fieldInstance, path);
                      path.remove(path.size()-1);
                  } catch (IllegalAccessException e) {
                      throw new AssertionError(e);
                  }
              });
    }

}
