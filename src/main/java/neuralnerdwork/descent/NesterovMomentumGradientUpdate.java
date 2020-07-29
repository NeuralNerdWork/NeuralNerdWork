package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public class NesterovMomentumGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double decayRate;
    private double[] momentum;

    public NesterovMomentumGradientUpdate(double learningRate, double decayRate) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        if (momentum == null) {
            momentum = new double[parameterBindings.size()];
        }

        final Model.ParameterBindings lookAheadBindings = parameterBindings.copy();
        {
            int i = 0;
            for (int variable : parameterBindings.variables()) {
                lookAheadBindings.put(variable, lookAheadBindings.get(i) + decayRate * momentum[i]);
                i++;
            }
        }
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);

        final double[] updatedMomentumComponents = new double[momentum.length];
        for (int i = 0; i < updatedMomentumComponents.length; i++) {
            updatedMomentumComponents[i] = decayRate * momentum[i] - learningRate * rawGradient.get(0, i);
        }
        momentum = updatedMomentumComponents;

        return momentum;
    }
}
