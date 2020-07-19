package neuralnerdwork.weight;

import java.util.Random;
import java.util.function.BiFunction;

public class VariableWeightInitializer {
    public static BiFunction<Integer, Integer, Double> dumbRandomWeightInitializer(Random rand) {
        return (r, c) -> (rand.nextDouble() - 0.5) * 2.0;
    } 

    public static BiFunction<Integer, Integer, Double> smartRandomWeightInitializer(Random rand) {
        return (r, c) -> (rand.nextDouble() - 0.5) * 2.0;
    } 

}