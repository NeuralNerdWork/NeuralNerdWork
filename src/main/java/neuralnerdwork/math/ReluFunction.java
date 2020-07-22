package neuralnerdwork.math;

import java.util.Random;

import neuralnerdwork.backprop.Layer;

public class ReluFunction implements ActivationFunction {
    @Override
    public String getFunctionName() {
        return "relu";
    }

    @Override
    public double apply(double input) {
        return Math.max(0, input);
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
                return 0;
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
        return (r.nextDouble() - 0.5) * (bound * 2);
    }
}
