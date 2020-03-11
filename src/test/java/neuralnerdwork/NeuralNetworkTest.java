package neuralnerdwork;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class NeuralNetworkTest {

    @Test
    void SimpleTestOfADummyNetwork() {
        NeuralNetwork network = new NeuralNetwork(List.of(
                i -> Arrays.stream(i).map(d -> d + 1.0).toArray(Double[]::new),
                i -> Arrays.stream(i).map(d -> d + 1.0).toArray(Double[]::new),
                i -> Arrays.stream(i).map(d -> d + 1.0).toArray(Double[]::new)
        ));

        assertArrayEquals(new Double[]{4.0}, network.apply(new Double[]{1.0}));
    }

    @Test
    void TestOfAHandCalculatedNetwork() {
        NeuralNetwork network = new NeuralNetwork(List.of(
            new NeuralNetworkLayer( List.of(
                new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid),
                new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid)
            )),
            new NeuralNetworkLayer( List.of(
                    new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid)
            ))
        ));

        assertEquals(0.7216, network.apply(new Double[]{2.0, 3.0})[0], 0.0009);
    }
}
