package neuralnerdwork;

import java.util.List;
import java.util.function.UnaryOperator;

public class NeuralNetwork implements UnaryOperator<Double[]> {
    private final List<UnaryOperator<Double[]>> layers;

    public NeuralNetwork(List<UnaryOperator<Double[]>> layers) {
        this.layers = layers;
    }


    public Double[] apply(Double[] input) {
        Double[] current = input;
        for(UnaryOperator<Double[]> layer : layers) {
            current = layer.apply(current);
        }

        return current;
    }
}
