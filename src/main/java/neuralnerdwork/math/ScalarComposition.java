package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarComposition(ScalarFunction... functions) implements ScalarFunction {

    public ScalarComposition {
        if (functions == null || functions.length == 0) {
            throw new IllegalArgumentException("cannot pass in empty or null functions");
        }
    }

    @Override
    public double apply(double[] input) {
        double curOutput = functions[0].apply(input);
        for (int i = 1; i < functions.length; i++) {
            final ScalarFunction f = functions[i];
            curOutput = f.apply(new double[]{curOutput});
        }

        return curOutput;
    }

    /*
     * Uses chain rule
     */
    @Override
    public ScalarFunction differentiate(int variableIndex) {
        // Chain rule works from outside function to inside
        final int last = functions.length-1;
        final ScalarFunction lastDerivative = functions[last].differentiate(variableIndex);
        final ScalarFunction[] composedFunctions = Arrays.copyOf(functions, last);
        final ScalarFunction[] composedWithOutsideDerivative = Arrays.copyOf(composedFunctions, last + 1);
        composedWithOutsideDerivative[last] = lastDerivative;

        // Chain Rule
        // (fog)'(x) = f'(g(x))*g'(x)
        return new ScalarMultiplicationFunction(
                new ScalarComposition(composedWithOutsideDerivative),
                new ScalarComposition(composedFunctions).differentiate(variableIndex)
        );
    }
}
