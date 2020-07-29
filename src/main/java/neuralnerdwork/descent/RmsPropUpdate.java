package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public class RmsPropUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double decayRate;
    private final double epsilon;

    /*
     There are exponentially decaying cumulative averages of squared values
     */
    private double[] gradientAverage;

    public RmsPropUpdate(double learningRate, double decayRate, double epsilon) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        if (gradientAverage == null) {
            gradientAverage = new double[parameterBindings.size()];
        }

        final double[] updateVectorValues = new double[rawGradient.getNumCols()];
        for (int i = 0; i < updateVectorValues.length; i++) {
            final double gradientComponent = rawGradient.get(0, i);
            gradientAverage[i] = gradientAverage[i] * decayRate + (1 - decayRate) * gradientComponent * gradientComponent;
            updateVectorValues[i] = -learningRate * (gradientComponent / (Math.sqrt(gradientAverage[i]) + epsilon));
        }

        return updateVectorValues;
    }
}
