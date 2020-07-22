package neuralnerdwork.math;

import java.util.Random;

import neuralnerdwork.backprop.Layer;

public class LeakyRelu implements ActivationFunction {

    private final double alpha;

    public LeakyRelu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public String getFunctionName() {
        return "leaky relu";
    }

    @Override
    public double apply(double input) {
        if (input >= 0.0) {
            return input;
        } else {
            return input * alpha;
        }
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new ReluDerivative();
    }

    private class ReluDerivative implements SingleVariableFunction {
        @Override
        public String getFunctionName() {
            return "relu derivative";
        }

        @Override
        public double apply(double input) {
            if (input < 0) {
                return alpha;
            } else {
                return 1;
            }
        }

        @Override
        public SingleVariableFunction differentiateByInput() {
            throw new UnsupportedOperationException("Not yet implemented!");
        }
    }

    @Override
    public double generateInitialWeight(Random r, Layer<?> layer) {
        double bound = Math.sqrt(2) * Math.sqrt(6.0 / (layer.inputLength() + layer.outputLength()));
        double nextDouble = r.nextDouble();
        double weight = (nextDouble - 0.5) * (bound * 2);
        return weight;
    }
}
