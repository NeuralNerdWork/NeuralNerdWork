package neuralnerdwork.descent;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

import java.util.Arrays;

public class AverageGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final Vector[] buffer;
    private double[] movingAverage;
    int curIndex = 0;

    public AverageGradientUpdate(double learningRate, int windowSize) {
        this.learningRate = learningRate;
        buffer = new Vector[windowSize];
    }

    @Override
    public Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final Vector rawGradient = error.computeDerivative(parameterBindings);
        if (movingAverage == null) {
            movingAverage = rawGradient.toArray();
            Arrays.fill(buffer, rawGradient);
        }

        final Vector toRemove = buffer[curIndex];
        buffer[curIndex] = rawGradient;
        curIndex = (curIndex + 1) % buffer.length;
        for (int i = 0; i < movingAverage.length; i++) {
            movingAverage[i] = movingAverage[i] + (rawGradient.get(i) - toRemove.get(i)) / buffer.length;
        }

        final double[] retVal = new double[movingAverage.length];
        for (int i = 0; i < retVal.length; i++) {
            retVal[i] = -learningRate * movingAverage[i];
        }

        return new ConstantVector(retVal);
    }
}
