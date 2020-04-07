package neuralnerdwork;

import neuralnerdwork.math.*;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MathTest {

    private static final VectorVariableBinding EMPTY_BINDING = new VectorVariableBinding(new VectorVariable(new ScalarVariable[0]), new ConstantVector(new double[0]));

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

        final Matrix result = multiplication.apply(EMPTY_BINDING);
        final MatrixEqualityComparator comparison = new MatrixEqualityComparator();

        final Matrix expected = other.apply(EMPTY_BINDING);
        assertTrue(comparison.equal(expected, result, 0.0001),
                   String.format("Multiplying by identity was not equal\nExpected\n%s\nObserved\n%s\n",
                                 expected,
                                 result));
    }

    @Test
    void twoMatrixPlusVectorDerivativeInOuterMatrix() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+b
        final ParameterMatrix w1 = new ParameterMatrix("w1", 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix("w2", 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
                new MatrixMultiplyFunction(
                        w2,
                        w1
                ),
                vector
        );

        final ScalarVariable variable = w2.variableFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variable);

        assertEquals(Set.of(
                "w1(0,0)",
                "w1(1,0)",
                "w1(0,1)",
                "w1(1,1)",
                "w2(0,0)",
                "w2(1,0)",
                "w2(0,1)",
                "w2(1,1)"
        ), multiplication.variables()
                         .stream()
                         .map(ScalarVariable::symbol)
                         .collect(Collectors.toSet()));

        assertArgumentInvariant(multiplication.variables(), valuesByVar -> {
            final Vector derivativeVector = derivative.apply(vectorBindingOf(valuesByVar));
            final double firstVar = lookupVariable("w1(1,0)", valuesByVar);
            final double secondVar = lookupVariable("w1(1,1)", valuesByVar);

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(firstVar - secondVar, derivativeVector.get(0), 0.0001);
            assertEquals(0.0, derivativeVector.get(1), 0.0001);
        });
    }

    @Test
    void twoMatrixPlusVectorDerivativeInInnerMatrix() {
        // https://www.wolframalpha.com/input/?i=derivative+of+%7B%7Ba%2Cb%7D%2C%7Bc%2Cd%7D%7D%7B%7Be%2Cf%7D%2C%7Bg%2Ch%7D%7D%7B%7B1%7D%2C%7B-1%7D%7D+by+f
        final ParameterMatrix w1 = new ParameterMatrix("w1", 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix("w2", 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
                new MatrixMultiplyFunction(
                        w2,
                        w1
                ),
                vector
        );

        final ScalarVariable variable = w1.variableFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variable);

        assertEquals(Set.of(
                "w1(0,0)",
                "w1(1,0)",
                "w1(0,1)",
                "w1(1,1)",
                "w2(0,0)",
                "w2(1,0)",
                "w2(0,1)",
                "w2(1,1)"
        ), multiplication.variables()
                         .stream()
                         .map(ScalarVariable::symbol)
                         .collect(Collectors.toSet()));

        assertArgumentInvariant(multiplication.variables(), valuesByVar -> {
            final VectorVariableBinding vectorVarBinding = vectorBindingOf(valuesByVar);
            final Vector derivativeVector = derivative.apply(vectorVarBinding);
            final double firstVar = lookupVariable("w2(0,0)", valuesByVar);
            final double secondVar = lookupVariable("w2(1,0)", valuesByVar);

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
        final ParameterMatrix w1 = new ParameterMatrix("w1", 2, 2);
        final ParameterMatrix w2 = new ParameterMatrix("w2", 2, 2);
        final VectorFunction vector = new ConstantVector(new double[] { 1.0, -1.0 });
        final MatrixVectorMultFunction multiplication = new MatrixVectorMultFunction(
                w2,
                new MatrixVectorMultFunction(
                        w1,
                        vector
                )
        );

        final ScalarVariable variable = w1.variableFor(0, 1);
        final VectorFunction derivative = multiplication.differentiate(variable);

        assertEquals(Set.of(
                "w1(0,0)",
                "w1(1,0)",
                "w1(0,1)",
                "w1(1,1)",
                "w2(0,0)",
                "w2(1,0)",
                "w2(0,1)",
                "w2(1,1)"
        ), multiplication.variables()
                         .stream()
                         .map(ScalarVariable::symbol)
                         .collect(Collectors.toSet()));

        assertArgumentInvariant(multiplication.variables(), valuesByVar -> {
            final VectorVariableBinding vectorVarBinding = vectorBindingOf(valuesByVar);
            final Vector derivativeVector = derivative.apply(vectorVarBinding);
            final double firstVar = lookupVariable("w2(0,0)", valuesByVar);
            final double secondVar = lookupVariable("w2(1,0)", valuesByVar);

            assertEquals(2, derivativeVector.length(), "Length not equal");
            assertEquals(-1.0 * firstVar, derivativeVector.get(0), 0.0001);
            assertEquals(-1.0 * secondVar, derivativeVector.get(1), 0.0001);
        });
    }

    private void assertArgumentInvariant(Set<ScalarVariable> variables, Consumer<List<ScalarVariableBinding>> assertions) {
        for (int attempt = 0; attempt < 3; attempt++) {
            final List<ScalarVariableBinding> valuesByVar = variables.stream()
                                                                     .map(var -> new ScalarVariableBinding(var, randomDouble()))
                                                                     .collect(toList());
            assertions.accept(valuesByVar);
        }

    }

    private double lookupVariable(String variableSymbol, List<ScalarVariableBinding> valuesByVar) {
        return valuesByVar.stream()
                          .filter(binding -> binding.variable().symbol().equals(variableSymbol))
                          .mapToDouble(ScalarVariableBinding::value)
                          .findFirst()
                          .orElseThrow(() -> new AssertionError("Couldn't find variable"));
    }

    private VectorVariableBinding vectorBindingOf(List<ScalarVariableBinding> valuesByVar) {
        final ScalarVariable[] scalarVars = valuesByVar.stream()
                                                       .map(ScalarVariableBinding::variable)
                                                       .toArray(ScalarVariable[]::new);

        final double[] inputValues = valuesByVar.stream()
                                                .mapToDouble(ScalarVariableBinding::value)
                                                .toArray();
        return new VectorVariableBinding(new VectorVariable(scalarVars), new ConstantVector(inputValues));
    }

}

    private double randomDouble() {
        return (Math.random() - 0.5) * 100.0;
    }
}
