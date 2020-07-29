package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import org.ejml.data.DMatrix;

import java.util.Arrays;

public class AverageGradientUpdate implements WeightUpdateStrategy {
    private final double learningRate;
    private final double[][] buffer;
    private double[] movingAverage;
    int curIndex = 0;

    public AverageGradientUpdate(double learningRate, int windowSize) {
        this.learningRate = learningRate;
        buffer = new double[windowSize][];
    }

    @Override
    public double[] updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings) {
        final DMatrix rawGradient = error.computeDerivative(parameterBindings);
        if (movingAverage == null) {
            movingAverage = new double[rawGradient.getNumCols()];
            Arrays.fill(buffer, new double[rawGradient.getNumCols()]);
        }

        final double[] toRemove = buffer[curIndex];
        for (int i = 0; i < rawGradient.getNumCols(); i++) {
            buffer[curIndex][i] = rawGradient.get(0, i);
        }
        curIndex = (curIndex + 1) % buffer.length;
        for (int i = 0; i < movingAverage.length; i++) {
            movingAverage[i] = movingAverage[i] + (rawGradient.get(0, i) - toRemove[i]) / buffer.length;
        }

        final double[] retVal = new double[movingAverage.length];
        for (int i = 0; i < retVal.length; i++) {
            retVal[i] = -learningRate * movingAverage[i];
        }

        return retVal;
    }
}
