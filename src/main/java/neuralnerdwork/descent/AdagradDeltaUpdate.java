package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

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
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final int[] variables = parameterBindings.variables();
        final Vector rawGradient = error.computeDerivative(parameterBindings, variables)
                                        .evaluate(parameterBindings);
        if (gradientAverage == null) {
            gradientAverage = new double[variables.length];
            updateAverage = new double[variables.length];
        }

        final double[] updateVectorValues = new double[rawGradient.length()];
        for (int i = 0; i < updateVectorValues.length; i++) {
            final double gradientComponent = rawGradient.get(i);
            gradientAverage[i] = gradientAverage[i] * decayRate + (1 - decayRate) * gradientComponent * gradientComponent;
            updateVectorValues[i] = -gradientComponent
                                    * ((Math.sqrt(updateAverage[i]) + epsilon)
                                       / (Math.sqrt(gradientAverage[i]) + epsilon));
            updateAverage[i] = updateAverage[i] * decayRate + (1 - decayRate) * updateVectorValues[i] * updateVectorValues[i];
        }

        return new ConstantVector(updateVectorValues);
    }
}
