package neuralnerdwork;

import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;

public class NeuralNetworkLayer implements UnaryOperator<Double[]> {
    private final List<Function<Double[], Double>> neurons;

    public NeuralNetworkLayer(List<Function<Double[], Double>> neurons) {
        this.neurons = neurons;
    }

    public Double[] apply(Double[] inputs) {
        return neurons.stream().map(n -> n.apply(inputs)).toArray(Double[]::new);
    }
}
