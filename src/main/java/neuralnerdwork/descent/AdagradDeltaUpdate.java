package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public class AdagradDeltaUpdate implements WeightUpdateStrategy {
    private final double decayRate;
    private final double epsilon;

    /*
     There are exponentially decaying cumulative averages of squared values
     */
    private double[] gradientAverage;
    private double[] updateAverage;

    public AdagradDeltaUpdate(double decayRate, double epsilon) {
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        if (gradientAverage == null) {
            gradientAverage = new double[parameterBindings.size()];
            updateAverage = new double[parameterBindings.size()];
        }

        final double[] updateVectorValues = new double[rawGradient.getNumCols()];
        for (int i = 0; i < updateVectorValues.length; i++) {
            final double gradientComponent = rawGradient.get(0, i);
            gradientAverage[i] = gradientAverage[i] * decayRate + (1 - decayRate) * gradientComponent * gradientComponent;
            updateVectorValues[i] = -gradientComponent
                                    * ((Math.sqrt(updateAverage[i]) + epsilon)
                                       / (Math.sqrt(gradientAverage[i]) + epsilon));
            updateAverage[i] = updateAverage[i] * decayRate + (1 - decayRate) * updateVectorValues[i] * updateVectorValues[i];
        }

        return updateVectorValues;
    }
}
