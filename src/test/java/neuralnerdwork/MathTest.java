package neuralnerdwork;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import neuralnerdwork.math.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opentest4j.AssertionFailedError;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static java.lang.String.format;
import static org.junit.jupiter.api.Assertions.*;

public class MathTest {

    private Random random;

    @BeforeEach
    public void setup() {
        final String seed = System.getProperty("neuralnerdwork.test.seed");
        if (seed != null) {
            random = new Random(Long.parseLong(seed));
        } else {
            random = new Random();
        }
    }

    @Test
    void multiplyByIdentityGivesSameMatrix() {
        final ConstantMatrix identity = new ConstantMatrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0}
        });


        MatrixExpression other = new ConstantMatrix(
                new double[][] {
                        { 2.0, -1.0 },
                        { 42.0, -1337 }
                }
        );

        final MatrixProduct multiplication = new MatrixProduct(
                identity,
                new MatrixProduct(
                        other,
                        identity
                )
        );

        final Matrix result = multiplication.evaluate(new Model().createBinder());
        final MatrixEqualityComparator comparison = new MatrixEqualityComparator();

        final Matrix expected = other.evaluate(new Model().createBinder());
        assertTrue(comparison.equal(expected, result, 0.0001),
                   format("Multiplying by identity was not equal\nExpected\n%s\nObserved\n%s\n",
                                 expected,
                                 result));
    }

    @Test
    void twoMatrixPlusVectorDerivativeInOuterMatrix() {
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
        final VectorExpression derivative = multiplication.computePartialDerivative(variable);

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }

            final Vector inputEval = vector.evaluate(binder);
            final Vector derivativeVector = derivative.evaluate(binder);
            final double firstVar = binder.get(w1.variableIndexFor(1, 0));
            final double secondVar = binder.get(w1.variableIndexFor(1, 1));

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(inputEval.get(0) * firstVar + inputEval.get(1) * secondVar, derivativeVector.get(0), 0.0001);
            assertEquals(0.0, derivativeVector.get(1), 0.0001);
        });
    }

    @Test
    void twoMatrixPlusVectorDerivativeInInnerMatrix() {
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
        final VectorExpression derivative = multiplication.computePartialDerivative(variable);

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }

            final Vector inputEval = vector.evaluate(binder);
            final Vector derivativeVector = derivative.evaluate(binder);
            final double firstVar = binder.get(w2.variableIndexFor(0, 0));
            final double secondVar = binder.get(w2.variableIndexFor(1, 0));

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(inputEval.get(1) * firstVar, derivativeVector.get(0), 0.0001);
            assertEquals(inputEval.get(1) * secondVar, derivativeVector.get(1), 0.0001);
        });
    }

    /*
     * Same as #twoMatrixPlusVectorDerivativeInInnerMatrix but with different multiplication object tree
     * to test associativity
     */
    @Test
    void associativityOfDerivative() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7Bi%7D%2C%7Bj%7D%7D+by+f
        final Model builder = new Model();
        final VectorExpression vector = builder.createParameterVector(2);
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
        final VectorExpression derivative = multiplication.computePartialDerivative(variable);

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }
            final Vector inputEval = vector.evaluate(binder);
            final Vector derivativeVector = derivative.evaluate(binder);
            final double firstVar = binder.get(w2.variableIndexFor(0, 0));
            final double secondVar = binder.get(w2.variableIndexFor(1, 0));

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(inputEval.get(1) * firstVar, derivativeVector.get(0), 0.0001);
            assertEquals(inputEval.get(1) * secondVar, derivativeVector.get(1), 0.0001);
        });
    }

    @Test
    void partialDerviativeWeightsWithActivation() {
        final Model builder = new Model();
        final VectorExpression vector = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                vector
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new SingleVariableLogisticFunction(),
                                                                                    weightedInputs);

        final VectorExpression partialDerivative = layerFunction.computePartialDerivative(w1.variableIndexFor(0, 0));
        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }
            final Vector observed = partialDerivative.evaluate(binder);

            assertEquals(2, observed.length(), "Length not equal");
            final Vector inputEval = vector.evaluate(binder);
            final double logisticInput = binder.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + binder.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
            final double firstExpected = logistic(logisticInput) * logistic(-logisticInput) * inputEval.get(0);
            assertEquals(firstExpected, observed.get(0), 0.0001);
            assertEquals(0.0, observed.get(1), 0.0001);
        });
    }

    @Test
    void fullDerivativeWeightsWithActivation() {
        final Model builder = new Model();
        final ParameterVector vector = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                vector
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new SingleVariableLogisticFunction(),
                                                                                    weightedInputs);

        final int[] weightVariables = Arrays.copyOfRange(builder.variables(), vector.variableStartIndex() + vector.length(), builder.size());
        final MatrixExpression derivative = layerFunction.computeDerivative(weightVariables);
        assertEquals(2, derivative.rows(), "Rows not equal");
        assertEquals(4, derivative.cols(), "Cols not equal");

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }
            final Matrix observed = derivative.evaluate(binder);

            assertEquals(2, observed.rows(), "Rows not equal");
            assertEquals(4, observed.cols(), "Cols not equal");

            final Vector inputEval = vector.evaluate(binder);
            final double firstLogisticInput = binder.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + binder.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
            final double secondLogisticInput = binder.get(w1.variableIndexFor(1, 0)) * inputEval.get(0) + binder.get(w1.variableIndexFor(1, 1)) * inputEval.get(1);

            final double firstExpected = logistic(firstLogisticInput) * logistic(-firstLogisticInput) * inputEval.get(0);
            final double secondExpected = logistic(firstLogisticInput) * logistic(-firstLogisticInput) * inputEval.get(1);
            final double thirdExpected = logistic(secondLogisticInput) * logistic(-secondLogisticInput) * inputEval.get(0);
            final double fourthExpected = logistic(secondLogisticInput) * logistic(-secondLogisticInput) * inputEval.get(1);

            assertEquals(firstExpected, observed.get(0, 0), 0.0001);
            assertEquals(0.0, observed.get(1, 0), 0.0001);

            assertEquals(secondExpected, observed.get(0, 1), 0.0001);
            assertEquals(0.0, observed.get(1, 1), 0.0001);

            assertEquals(0.0, observed.get(0, 2), 0.0001);
            assertEquals(thirdExpected, observed.get(1, 2), 0.0001);

            assertEquals(0.0, observed.get(0, 3), 0.0001);
            assertEquals(fourthExpected, observed.get(1, 3), 0.0001);
        });
    }

    @Test
    void fullDerivativeOfLossFunctionOnLayer() {
        final Model builder = new Model();
        final ParameterVector trainingInput = builder.createParameterVector(2);
        final ParameterMatrix w1 = builder.createParameterMatrix(2, 2);
        final MatrixVectorProduct weightedInputs = new MatrixVectorProduct(
                w1,
                trainingInput
        );
        final VectorExpression layerFunction = new VectorizedSingleVariableFunction(new SingleVariableLogisticFunction(),
                                                                                    weightedInputs);

        final ConstantVector trainingOutput = new ConstantVector(new double[] { 1.0, 0.0 });
        final VectorSum error = new VectorSum(layerFunction, new ScaledVector(-1.0, trainingOutput));
        // TODO should make a square combinator; this will differentiate error twice
        final DotProduct squaredError = new DotProduct(error, error);

        final int[] weightVariables = Arrays.copyOfRange(builder.variables(), trainingInput.variableStartIndex() + trainingInput.length(), builder.size());
        final VectorExpression derivative = squaredError.computeDerivative(weightVariables);
        assertEquals(builder.size() - trainingInput.length(), derivative.length(), "Length not equal");

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }
            final Vector observed = derivative.evaluate(binder);

            assertEquals(4, observed.length(), "Length not equal");

            final Vector inputEval = trainingInput.evaluate(binder);
            final double unactivatedNeuron1 = binder.get(w1.variableIndexFor(0, 0)) * inputEval.get(0) + binder.get(w1.variableIndexFor(0, 1)) * inputEval.get(1);
            final double unactivatedNeuron2 = binder.get(w1.variableIndexFor(1, 0)) * inputEval.get(0) + binder.get(w1.variableIndexFor(1, 1)) * inputEval.get(1);

            final double[] expected = new double[] {
                    2.0 * (logistic(unactivatedNeuron1) - trainingOutput.get(0)) * logisticDerivative(unactivatedNeuron1) * inputEval.get(0),
                    2.0 * (logistic(unactivatedNeuron1) - trainingOutput.get(0)) * logisticDerivative(unactivatedNeuron1) * inputEval.get(1),
                    2.0 * (logistic(unactivatedNeuron2) - trainingOutput.get(1)) * logisticDerivative(unactivatedNeuron2) * inputEval.get(0),
                    2.0 * (logistic(unactivatedNeuron2) - trainingOutput.get(1)) * logisticDerivative(unactivatedNeuron2) * inputEval.get(1)
            };

            assertArrayEquals(expected, observed.toArray(), 0.0001, format("Expected: %s, Observed: %s", Arrays.toString(expected), Arrays.toString(observed.toArray())));
        });
    }

    @Test
    void noZeroValuesInDerivative() {
        final int rows = 10;
        final int cols = 10;
        final int numLayers = 5;

        final Model builder = new Model();
        final VectorExpression trainingInput = builder.createParameterVector(cols);

        final SingleVariableLogisticFunction logistic = new SingleVariableLogisticFunction();

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

        final ConstantVector trainingOutput = new ConstantVector(randomDoubles(rows));
        final VectorSum error = new VectorSum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                                                             new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final VectorExpression lossDerivative = logTiming("Computed derivative function", () -> squaredError.computeDerivative(builder.variables()));
        assertNonZero(lossDerivative);
    }

    @Test
    void derivativeOfManyLayers() {
        final int rows = 10;
        final int cols = 10;
        final int numLayers = 50;

        final Model builder = new Model();
        final VectorExpression trainingInput = builder.createParameterVector(cols);

        final SingleVariableLogisticFunction logistic = new SingleVariableLogisticFunction();

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

        final ConstantVector trainingOutput = new ConstantVector(randomDoubles(rows));
        final VectorSum error = new VectorSum(networkFunction, new ScaledVector(-1.0, trainingOutput));

        final double[] ones = new double[rows];
        Arrays.fill(ones, 1.0);
        final ScalarExpression squaredError = new DotProduct(new ConstantVector(ones),
                                                             new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), error));

        final VectorExpression lossDerivative = logTiming("Computed derivative function", () -> squaredError.computeDerivative(builder.variables()));

        assertArgumentInvariant(builder.size(), values -> {
            final Model.Binder binder = builder.createBinder();
            final int[] vars = binder.variables();
            for (int i = 0; i < vars.length; i++) {
                binder.put(vars[i], values[i]);
            }
            System.out.println();
            logTiming("Invoked undifferentiated error", () -> squaredError.evaluate(binder));
            logTiming("Printed derivative", () -> {
                try {
                    final File tmpFile = File.createTempFile("derivative", ".json");
                    System.out.println("Writing to tmp file " + tmpFile.getAbsolutePath());
                    new ObjectMapper()
                            .setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE)
                            .setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY)
                    .writeValue(tmpFile, lossDerivative);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                return null;
            });
            logTiming("Applied derivative function", () -> lossDerivative.evaluate(binder));
            System.out.println();
        });
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

    private <T> T logTiming(String actionName, Supplier<T> action) {
        final Instant start = Instant.now();
        final T retVal = action.get();
        final Duration duration = Duration.between(start, Instant.now());
        System.out.printf("%s function in %d ms\n", actionName, duration.toMillis());
        return retVal;
    }

    private double logisticDerivative(double x) {
        return logistic(x) * logistic(-x);
    }

    private double logistic(double x) {
        return Math.exp(x) / (1 + Math.exp(x));
    }

    private void assertArgumentInvariant(int length, Consumer<double[]> assertions) {
        for (int attempt = 0; attempt < 3; attempt++) {
            final double[] values = randomDoubles(length);
            try {
                assertions.accept(values);
            } catch (Exception | AssertionFailedError e) {
                throw new AssertionError("Failed on inputs: " + Arrays.toString(values), e);
            }
        }

    }

    private double[] randomDoubles(int length) {
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = randomDouble();
        }
        return values;
    }

    private double randomDouble() {
        return (random.nextDouble() - 0.5) * 10.0;
    }
}
