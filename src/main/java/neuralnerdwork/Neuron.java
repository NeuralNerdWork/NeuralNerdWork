package neuralnerdwork;

import java.util.function.Function;

public class Neuron implements Function<Double[], Double> {
    public final Double bias;
    public final Double[] weights;
    public final Function<Double, Double> activation;

    public Neuron(Double bias, Double[] weights,
            Function<Double, Double> activation) {
        this.bias = bias;
        this.weights = weights;
        this.activation = activation;
    }

    public Double apply(Double[] inputs){
        assert inputs.length == weights.length :
                "A Neuron can only process the same number of inputs as it has weights";

        Double total = bias;
        for(int i = 0; i < weights.length; i++) {
            total += inputs[i] * weights[i];
        }
        return activation.apply(total);
    }
}
