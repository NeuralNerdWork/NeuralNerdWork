package neuralnerdwork.math;

import neuralnerdwork.backprop.Layer;
import neuralnerdwork.weight.VariableWeightInitializer;

import java.util.Random;

public class IdentityFunction implements ActivationFunction {
    @Override
    public String getFunctionName() {
        return "identity";
    }

    @Override
    public double apply(double input) {
        return input;
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new IdentityDerivative();
    }

    private class IdentityDerivative implements SingleVariableFunction {
        @Override
        public String getFunctionName() {
            return "identity derivative";
        }

        @Override
        public double apply(double input) {
            return 1;
        }

        @Override
        public SingleVariableFunction differentiateByInput() {
            throw new UnsupportedOperationException("Not yet implemented!");
        }
    }

    @Override
    public double generateInitialWeight(Random r, Layer<?> layer) {
        return VariableWeightInitializer.dumbRandomWeightInitializer(r).apply(layer);
    }
}
