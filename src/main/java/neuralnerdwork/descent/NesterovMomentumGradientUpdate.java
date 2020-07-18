package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

import java.util.Arrays;

public class NesterovMomentumGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double decayRate;
    private Vector momentum;

    public NesterovMomentumGradientUpdate(double learningRate, double decayRate) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
    }

    @Override
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        if (momentum == null) {
            final Vector initialGradient = error.computeDerivative(parameterBindings, parameterBindings.variables());
            momentum = new ConstantVector(
                    Arrays.stream(initialGradient.toArray())
                          .map(x -> -x)
                          .toArray()
            );
        }

        final Model.ParameterBindings lookAheadBindings = parameterBindings.copy();
        final int[] variables = lookAheadBindings.variables();
        for (int i = 0; i < variables.length; i++) {
            lookAheadBindings.put(variables[i], lookAheadBindings.get(i) + decayRate * momentum.get(i));
        }
        final Vector rawGradient = error.computeDerivative(parameterBindings, variables);

        final double[] updatedMomentumComponents = new double[momentum.length()];
        for (int i = 0; i < updatedMomentumComponents.length; i++) {
            updatedMomentumComponents[i] = decayRate * momentum.get(i) - learningRate * rawGradient.get(i);
        }
        momentum = new ConstantVector(updatedMomentumComponents);

        return momentum;
    }
}
