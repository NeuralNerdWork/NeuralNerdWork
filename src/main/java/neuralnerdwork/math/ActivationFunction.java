package neuralnerdwork.math;

import java.util.Random;

import neuralnerdwork.backprop.Layer;

public interface ActivationFunction extends SingleVariableFunction {
    double generateInitialWeight(Random r, Layer<?> layer);
}