package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Set;

public record ScalarComposition(ScalarFunction... functions) implements ScalarFunction {

    public ScalarComposition {
        if (functions == null || functions.length == 0) {
            throw new IllegalArgumentException("cannot pass in empty or null functions");
        }
        for (int i = 1; i < functions.length; i++) {
            if (functions[i].variables().size() > 1) {
                throw new IllegalArgumentException("only the first function can have more than one argument");
            }
        }
    }

    @Override
    public double apply(VectorVariableBinding input) {
        double curOutput = functions[0].apply(input);
        for (int i = 1; i < functions.length; i++) {
            final ScalarFunction f = functions[i];
            final ScalarVariable variable = f.variables().iterator().next();
            curOutput = f.apply(new VectorVariableBinding(new VectorVariable(new ScalarVariable[]{variable}), new ConstantVector(new double[]{curOutput})));
        }

        return curOutput;
    }

    /*
     * Uses chain rule
     */
    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        // Chain rule works from outside function to inside
        final int last = functions.length-1;
        final ScalarFunction lastDerivative = functions[last].differentiate(variable);
        final ScalarFunction[] composedFunctions = Arrays.copyOf(functions, last);
        final ScalarFunction[] composedWithOutsideDerivative = Arrays.copyOf(composedFunctions, last + 1);
        composedWithOutsideDerivative[last] = lastDerivative;

        // Chain Rule
        // (fog)'(x) = f'(g(x))*g'(x)
        return new ScalarMultiplicationFunction(
                new ScalarComposition(composedWithOutsideDerivative),
                new ScalarComposition(composedFunctions).differentiate(variable)
        );
    }

    @Override
    public VectorFunction differentiate(VectorVariable variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return functions[0].variables();
    }
}
