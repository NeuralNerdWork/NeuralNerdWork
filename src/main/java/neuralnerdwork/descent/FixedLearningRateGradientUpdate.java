package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

public record FixedLearningRateGradientUpdate(double learningRate) implements WeightUpdateStrategy {
    @Override
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final Vector rawGradient = error.computeDerivative(parameterBindings);
        final double[] updateValues = rawGradient.toArray();
        for (int i = 0; i < updateValues.length; i++) {
            updateValues[i] *= -learningRate;
        }

        return new ConstantVector(updateValues);
    }
}
