package neuralnerdwork;

import neuralnerdwork.math.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Consumer;

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


        MatrixFunction other = new ConstantMatrix(
                new double[][] {
                        { 2.0, -1.0 },
                        { 42.0, -1337 }
                }
        );

        final MatrixMultiplyFunction multiplication = new MatrixMultiplyFunction(
                identity,
                new MatrixMultiplyFunction(
                        other,
                        identity
                )
        );

        final Matrix result = multiplication.apply(new double[0]);
        final MatrixEqualityComparator comparison = new MatrixEqualityComparator();

        final Matrix expected = other.apply(new double[0]);
        assertTrue(comparison.equal(expected, result, 0.0001),
                   format("Multiplying by identity was not equal\nExpected\n%s\nObserved\n%s\n",
                                 expected,
                                 result));
    }

    @Test
    void twoMatrixPlusVectorDerivativeInOuterMatrix() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+b
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix(4, 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction multiplication = new MatrixVectorProductFunction(
                new MatrixMultiplyFunction(
                        w2,
                        w1
                ),
                vector
        );

        final int variableIndex = w2.indexFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variableIndex);

        assertArgumentInvariant(8, values -> {
            final Vector derivativeVector = derivative.apply(values);
            final double firstVar = values[w1.indexFor(1, 0)];
            final double secondVar = values[w1.indexFor(1, 1)];

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(firstVar - secondVar, derivativeVector.get(0), 0.0001);
            assertEquals(0.0, derivativeVector.get(1), 0.0001);
        });
    }

    @Test
    void twoMatrixPlusVectorDerivativeInInnerMatrix() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+f
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix(4, 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction multiplication = new MatrixVectorProductFunction(
                new MatrixMultiplyFunction(
                        w2,
                        w1
                ),
                vector
        );

        final int variableIndex = w1.indexFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variableIndex);

        assertArgumentInvariant(8, values -> {
            final Vector derivativeVector = derivative.apply(values);
            final double firstVar = values[w2.indexFor(0, 0)];
            final double secondVar = values[w2.indexFor(1, 0)];

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(-1.0 * firstVar, derivativeVector.get(0), 0.0001);
            assertEquals(-1.0 * secondVar, derivativeVector.get(1), 0.0001);
        });
    }

    /*
     * Same as #twoMatrixPlusVectorDerivativeInInnerMatrix but with different multiplication object tree
     * to test associativity
     */
    @Test
    void associativityOfDerivative() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+f
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix(4, 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction multiplication = new MatrixVectorProductFunction(
                w2,
                new MatrixVectorProductFunction(
                        w1,
                        vector
                )
        );

        final int variableIndex = w1.indexFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variableIndex);

        assertArgumentInvariant(8, values -> {
            final Vector derivativeVector = derivative.apply(values);
            final double firstVar = values[w2.indexFor(0, 0)];
            final double secondVar = values[w2.indexFor(1, 0)];

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(-1.0 * firstVar, derivativeVector.get(0), 0.0001);
            assertEquals(-1.0 * secondVar, derivativeVector.get(1), 0.0001);
        });
    }

    @Test
    void partialDerviativeWeightsWithActivation() {
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ConstantVector vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction weightedInputs = new MatrixVectorProductFunction(
                w1,
                vector
        );
        final SingleVariableLogisticFunction logistic = new SingleVariableLogisticFunction();
        final VectorizedSingleVariableFunctions activationFunction = new VectorizedSingleVariableFunctions(logistic, logistic);
        final VectorFunctionComposition layerFunction = new VectorFunctionComposition(weightedInputs, activationFunction);

        final VectorFunction partialDerivative = layerFunction.differentiate(w1.indexFor(0, 0));
        assertArgumentInvariant(4, values -> {
            final Vector observed = partialDerivative.apply(values);

            assertEquals(2, observed.length(), "Length not equal");
            final double logisticInput = values[0] * vector.get(w1.indexFor(0, 0)) + values[w1.indexFor(0, 1)] * vector.get(1);
            final double firstExpected = logistic(logisticInput) * logistic(-logisticInput) * vector.get(0);
            assertEquals(firstExpected, observed.get(0), 0.0001);
            assertEquals(0.0, observed.get(1), 0.0001);
        });
    }

    @Test
    void fullDerivativeWeightsWithActivation() {
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ConstantVector vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction weightedInputs = new MatrixVectorProductFunction(
                w1,
                vector
        );
        final SingleVariableLogisticFunction logistic = new SingleVariableLogisticFunction();
        final VectorizedSingleVariableFunctions activationFunction = new VectorizedSingleVariableFunctions(logistic, logistic);
        final VectorFunctionComposition layerFunction = new VectorFunctionComposition(weightedInputs, activationFunction);

        final MatrixFunction derivative = layerFunction.differentiate();
        assertEquals(2, derivative.rows(), "Rows not equal");
        assertEquals(4, derivative.cols(), "Cols not equal");

        assertArgumentInvariant(4, values -> {
            final Matrix observed = derivative.apply(values);

            assertEquals(2, observed.rows(), "Rows not equal");
            assertEquals(4, observed.cols(), "Cols not equal");

            final double firstLogisticInput = values[w1.indexFor(0, 0)] * vector.get(0) + values[w1.indexFor(0, 1)] * vector.get(1);
            final double secondLogisticInput = values[w1.indexFor(1, 0)] * vector.get(0) + values[w1.indexFor(1, 1)] * vector.get(1);

            final double firstExpected = logistic(firstLogisticInput) * logistic(-firstLogisticInput) * vector.get(0);
            final double secondExpected = logistic(firstLogisticInput) * logistic(-firstLogisticInput) * vector.get(1);
            final double thirdExpected = logistic(secondLogisticInput) * logistic(-secondLogisticInput) * vector.get(0);
            final double fourthExpected = logistic(secondLogisticInput) * logistic(-secondLogisticInput) * vector.get(1);

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
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ConstantVector trainingInput = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorProductFunction weightedInputs = new MatrixVectorProductFunction(
                w1,
                trainingInput
        );
        final SingleVariableLogisticFunction logistic = new SingleVariableLogisticFunction();
        final VectorizedSingleVariableFunctions activationFunction = new VectorizedSingleVariableFunctions(logistic, logistic);
        final VectorFunctionComposition layerFunction = new VectorFunctionComposition(weightedInputs, activationFunction);

        // TODO should make a scalar multiple combinator
        final ConstantVector negatedTrainingOutput = new ConstantVector(new double[] { -1.0, 0.0 });
        final VectorSumFunction error = new VectorSumFunction(layerFunction, negatedTrainingOutput);
        // TODO should make a square combinator; this will differentiate error twice
        final DotProduct squaredError = new DotProduct(error, error);

        final VectorFunction derivative = squaredError.differentiate();
        assertEquals(4, derivative.length(), "Length not equal");

        assertArgumentInvariant(4, values -> {
            final Vector observed = derivative.apply(values);

            assertEquals(4, observed.length(), "Length not equal");

            final double unactivatedNeuron1 = values[w1.indexFor(0, 0)] * trainingInput.get(0) + values[w1.indexFor(0, 1)] * trainingInput.get(1);
            final double unactivatedNeuron2 = values[w1.indexFor(1, 0)] * trainingInput.get(0) + values[w1.indexFor(1, 1)] * trainingInput.get(1);

            final double[] expected = new double[] {
                    2.0 * (logistic(unactivatedNeuron1) + negatedTrainingOutput.get(0)) * logisticDerivative(unactivatedNeuron1) * trainingInput.get(0),
                    2.0 * (logistic(unactivatedNeuron1) + negatedTrainingOutput.get(0)) * logisticDerivative(unactivatedNeuron1) * trainingInput.get(1),
                    2.0 * (logistic(unactivatedNeuron2) + negatedTrainingOutput.get(1)) * logisticDerivative(unactivatedNeuron2) * trainingInput.get(0),
                    2.0 * (logistic(unactivatedNeuron2) + negatedTrainingOutput.get(1)) * logisticDerivative(unactivatedNeuron2) * trainingInput.get(1)
            };

            assertArrayEquals(expected, observed.toArray(), 0.0001, format("Expected: %s, Observed: %s", Arrays.toString(expected), Arrays.toString(observed.toArray())));
        });
    }

    private double logisticDerivative(double x) {
        return logistic(x) * logistic(-x);
    }

    private double logistic(double x) {
        return Math.exp(x) / (1 + Math.exp(x));
    }

    private void assertArgumentInvariant(int length, Consumer<double[]> assertions) {
        for (int attempt = 0; attempt < 3; attempt++) {
            final double[] values = new double[length];
            for (int i = 0; i < length; i++) {
                values[i] = randomDouble();
            }
            assertions.accept(values);
        }

    }

    private double randomDouble() {
        return (random.nextDouble() - 0.5) * 100.0;
    }
}
