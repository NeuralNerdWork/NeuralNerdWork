package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

public class AdagradUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double epsilon;
    private double[] sumsOfSquares;

    public AdagradUpdate(double learningRate, double epsilon) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    @Override
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final int[] variables = parameterBindings.variables();
        final Vector rawGradient = error.computeDerivative(parameterBindings, variables);
        if (sumsOfSquares == null) {
            sumsOfSquares = new double[variables.length];
        }

        final double[] updateVectorValues = new double[rawGradient.length()];
        for (int i = 0; i < updateVectorValues.length; i++) {
            final double gradientComponent = rawGradient.get(i);
            sumsOfSquares[i] += gradientComponent * gradientComponent;
            updateVectorValues[i] = -learningRate * (gradientComponent / (Math.sqrt(sumsOfSquares[i]) + epsilon));
        }

        return new ConstantVector(updateVectorValues);
    }
}
