package neuralnerdwork;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import neuralnerdwork.math.ConstantVector;

public class LinearRegressionTest {
    @Test
    void testBasicLinearRegressionTraining() {

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 1}, 
            0.1, 
            () -> (Math.random()-0.5)*2.0
        );

        NeuralNetwork network = trainer.train(Arrays.asList(
            new TrainingSample(new ConstantVector(new double[]{0.0, 0.1}), new ConstantVector(new double[]{0.0})),
            new TrainingSample(new ConstantVector(new double[]{0.0, 1.3}), new ConstantVector(new double[]{1.0}))
        ));

        
        assertArrayEquals(new double[]{0.0}, network.apply(new double[]{0.0, 0.1}), 0.2);
        assertArrayEquals(new double[]{0.0}, network.apply(new double[]{0.0, 0.2}), 0.2);
        assertArrayEquals(new double[]{1.0}, network.apply(new double[]{0.0, 1.3}), 0.2);
        assertArrayEquals(new double[]{1.0}, network.apply(new double[]{0.0, 1.2}), 0.2);
    }
}
