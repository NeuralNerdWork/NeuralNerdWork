package neuralnerdwork;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class NeuralNetworkLayerTest {

    @Test
    void TestBasicLayer() {
        NeuralNetworkLayer layer = new NeuralNetworkLayer(List.of(
                inputs -> Arrays.stream(inputs).reduce(0.0, Double::sum),
                inputs -> Arrays.stream(inputs).reduce(0.0, Double::sum),
                inputs -> Arrays.stream(inputs).reduce(0.0, Double::sum),
                inputs -> Arrays.stream(inputs).reduce(0.0, Double::sum)
        ));

        assertArrayEquals(
                new Double[]{3.0,3.0,3.0,3.0}, layer.apply(new Double[]{1.0, 2.0}));
        }
}
