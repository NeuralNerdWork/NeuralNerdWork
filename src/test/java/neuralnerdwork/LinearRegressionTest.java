package neuralnerdwork;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;

public class LinearRegressionTest {
    @Test
    @Disabled
    void testBasicLinearRegressionTraining() {
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer();
        NeuralNetwork network = trainer.train(
                List.of(
                        new TrainingSample(new Double[]{0.0, 0.1}, new Double[]{0.0}),
                        new TrainingSample(new Double[]{0.0, 1.3}, new Double[]{1.0})
                ));

        assertArrayEquals(new Double[]{0.0}, network.apply(new Double[]{0.0, 0.1}));
        assertArrayEquals(new Double[]{1.0}, network.apply(new Double[]{0.0, 1.3}));
    }
}
