package neuralnerdwork;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;

class NeuronTest {

    @Test
    void processPairOfInputs() {
        Neuron neuron =
                new Neuron(
                        4.0,
                        new Double[]{0.0, 1.0},
                        Calculator::sigmoid);

        assertEquals(0.999, neuron.apply(new Double[]{2.0, 3.0}), 0.00009);
    }

    @Test
    void failToProcessMismatchedInputsAndWeights() {
        Neuron neuron =
                new Neuron(
                        4.0,
                        new Double[]{0.0, 1.0},
                        Calculator::sigmoid);
        assertThrows(AssertionError.class,
                     () -> neuron.apply(new Double[]{2.0, 3.0, 4.0}));
    }
}
