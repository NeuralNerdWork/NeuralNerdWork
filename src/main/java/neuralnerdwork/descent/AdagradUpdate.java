package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

public class AdagradUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double epsilon;
    private double[] sumsOfSquares;

    public AdagradUpdate(double learningRate, double epsilon) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        if (sumsOfSquares == null) {
            sumsOfSquares = new double[parameterBindings.size()];
        }

        final double[] updateVectorValues = new double[rawGradient.getNumCols()];
        for (int i = 0; i < updateVectorValues.length; i++) {
            final double gradientComponent = rawGradient.get(0, i);
            sumsOfSquares[i] += gradientComponent * gradientComponent;
            updateVectorValues[i] = -learningRate * (gradientComponent / (Math.sqrt(sumsOfSquares[i]) + epsilon));
        }

        return updateVectorValues;
    }
}
