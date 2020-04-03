package neuralnerdwork;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.List;


public class LossFunctionTest {
    @Test
    void TestBasicLayer() {
        NeuralNetwork network = new NeuralNetwork(List.of(
            new NeuralNetworkLayer( List.of(
                new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid),
                new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid)
            )),
            new NeuralNetworkLayer( List.of(
                    new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid),
                    new Neuron(0.0, new Double[]{0.0, 1.0}, Calculator::sigmoid)
            ))
        ));
        
        LossFunction lossFunction = new LossFunction(network);

        var inputs = new Double[][] {
            { 2.0, 3.0 }
        };
        var trueValues = new Double[][] { { 1.0, 1.0 } };

        double observedMse = lossFunction.evaluate(trueValues, inputs);

        var distance = Math.pow(Math.pow(1.0-0.7216, 2) + Math.pow(1.0-0.7216, 2), 0.5); // distance between vectors [1,1] and [0.7216,0.7216]
        double expectedMse = Math.pow(distance, 2.0) / 1.0;

        assertEquals(expectedMse, observedMse, 0.0009);
    }
}