package neuralnerdwork;

import neuralnerdwork.math.*;
import org.junit.jupiter.api.Test;

import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MathTest {

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
                   String.format("Multiplying by identity was not equal\nExpected\n%s\nObserved\n%s\n",
                                 expected,
                                 result));
    }

    @Test
    void twoMatrixPlusVectorDerivativeInOuterMatrix() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+b
        final ParameterMatrix w1 = new ParameterMatrix(0, 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix(4, 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
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
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
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
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
                w2,
                new MatrixVectorMultFunction(
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
        return (Math.random() - 0.5) * 100.0;
    }
}
