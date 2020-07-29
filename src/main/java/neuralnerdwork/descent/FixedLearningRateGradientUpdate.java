package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public record FixedLearningRateGradientUpdate(double learningRate) implements WeightUpdateStrategy {
    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        final double[] updateValues = new double[rawGradient.getNumCols()];
        for (int i = 0; i < updateValues.length; i++) {
            updateValues[i] = -learningRate * rawGradient.get(0, i);
        }

        return updateValues;
    }
}
