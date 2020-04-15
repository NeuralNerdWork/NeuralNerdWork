package neuralnerdwork.math;

import java.util.Arrays;

public record SingleVariableComposition(SingleVariableFunction... functions) implements SingleVariableFunction {

    public SingleVariableComposition {
        if (functions == null || functions.length == 0) {
            throw new IllegalArgumentException("cannot pass in empty or null functions");
        }
    }

    @Override
    public double apply(double input) {
        double curOutput = functions[0].apply(input);
        for (int i = 1; i < functions.length; i++) {
            final SingleVariableFunction f = functions[i];
            curOutput = f.apply(curOutput);
        }

        return curOutput;
    }

    /*
     * Uses chain rule
     */
    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        // Chain rule works from outside function to inside
        final int last = functions.length-1;
        final SingleVariableFunction lastDerivative = functions[last].differentiateBySingleVariable();
        final SingleVariableFunction[] composedFunctions = Arrays.copyOf(functions, last);
        final SingleVariableFunction[] composedWithOutsideDerivative = Arrays.copyOf(composedFunctions, last + 1);
        composedWithOutsideDerivative[last] = lastDerivative;

        // Chain Rule
        // (fog)'(x) = f'(g(x))*g'(x)
        return new SingleVariableProductFunction(
                new SingleVariableComposition(composedWithOutsideDerivative),
                new SingleVariableComposition(composedFunctions).differentiateBySingleVariable()
        );
    }
}
