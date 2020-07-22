package neuralnerdwork.weight;

import java.util.Random;
import java.util.function.Function;

import neuralnerdwork.backprop.Layer;

public class VariableWeightInitializer {
    public static Function<Layer<?>, Double> dumbRandomWeightInitializer(Random rand) {
        return (l) -> (rand.nextDouble() - 0.5) * 2.0;
    } 

    public static Function<Layer<?>, Double> smartRandomWeightInitializer(Random rand) {
        return (layer) -> layer.activation().generateInitialWeight(rand, layer);
        // return (l) -> (rand.nextDouble() - 0.5) * 2.0;
    } 
}