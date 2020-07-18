package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

import java.util.Arrays;

public class MomentumGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double decayRate;
    private Vector momentum;

    public MomentumGradientUpdate(double learningRate, double decayRate) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
    }

    @Override
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final Vector rawGradient = error.computeDerivative(parameterBindings, parameterBindings.variables())
                                        .evaluate(parameterBindings);
        if (momentum == null) {
            momentum = new ConstantVector(
                    Arrays.stream(rawGradient.toArray())
                          .map(x -> -x)
                          .toArray()
            );
        }

        final double[] updatedMomentumComponents = new double[momentum.length()];
        for (int i = 0; i < updatedMomentumComponents.length; i++) {
            updatedMomentumComponents[i] = decayRate * momentum.get(i) - learningRate * rawGradient.get(i);
        }
        momentum = new ConstantVector(updatedMomentumComponents);

        return momentum;
    }
}
