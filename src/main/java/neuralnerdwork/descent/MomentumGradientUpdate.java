package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public class MomentumGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double decayRate;
    private double[] momentum;

    public MomentumGradientUpdate(double learningRate, double decayRate) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        if (momentum == null) {
            momentum = new double[rawGradient.getNumCols()];
        }

        final double[] updatedMomentumComponents = new double[momentum.length];
        for (int i = 0; i < updatedMomentumComponents.length; i++) {
            updatedMomentumComponents[i] = decayRate * momentum[i] - learningRate * rawGradient.get(0, i);
        }
        momentum = updatedMomentumComponents;

        return momentum;
    }
}
