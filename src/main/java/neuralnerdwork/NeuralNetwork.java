package neuralnerdwork;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.Vector;

@FunctionalInterface
public interface NeuralNetwork {

    double[] apply(double[] input);
    default Vector apply(Vector input) {
        return new ConstantVector(apply(input.toArray()));
    }

}