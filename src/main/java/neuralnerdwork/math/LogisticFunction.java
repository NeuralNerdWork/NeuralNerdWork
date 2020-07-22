package neuralnerdwork.math;

import java.util.Random;

import neuralnerdwork.backprop.Layer;

public class LogisticFunction implements ActivationFunction {
    @Override
    public String getFunctionName() {
        return "logistic";
    }

    @Override
    public double apply(double input) {
        return logistic(input);
    }

    private static double logistic(double input) {
        final double exp = Math.exp(-input);
        if (Double.isInfinite(exp)) {
            return 0;
        } else {
            return 1 / (1 + exp);
        }
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new SingleVariableFunction() {
            @Override
            public String getFunctionName() {
                return "logistic derivative";
            }

            @Override
            public double apply(double input) {
                double logistic = logistic(input);
                return logistic * (1 - logistic);
            }

            @Override
            public SingleVariableFunction differentiateByInput() {
                throw new UnsupportedOperationException("Not implemented");
            }
        };
    }

    @Override
    public double generateInitialWeight(Random r, Layer<?> layer) {
        double bound = Math.sqrt(6.0 / (layer.inputLength() + layer.outputLength()));
        return (r.nextDouble() - 0.5) * (bound * 2);
    }
}
