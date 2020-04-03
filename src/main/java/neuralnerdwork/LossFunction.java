package neuralnerdwork;

import java.util.List;

public class LossFunction {

    private NeuralNetwork network;

    public LossFunction(NeuralNetwork network) {
        this.network = network;
    }

    public double evaluate(Double[][] trueValues, Double[][] inputs) {
        double accum = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            Double[] predictions = network.apply(inputs[i]);
            double distance2 = 0.0;
            for (int j = 0; j < trueValues[i].length; j++) {
                distance2 += Math.pow(trueValues[i][j] - predictions[j], 2.0);
            }
            // Normally you square-root to get the norm between vectors, but we are doing MSE so it cancels out
            accum += distance2;
        }

        return accum / trueValues.length;
    }
}