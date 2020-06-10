package neuralnerdwork;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.swing.JFrame;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.BasicStroke;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import neuralnerdwork.descent.SimpleBatchGradientDescent;
import neuralnerdwork.math.ConstantVector;

public class SimpleTrainingTest {
    @Test
    void trainingTwoLayerNetworkShouldConverge() {

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 1}, 
            new SimpleBatchGradientDescent(
                new SimpleBatchGradientDescent.HyperParameters(
                        0.1,
                        0.001,
                        1000
                ),
                () -> (Math.random() - 0.5) * 2.0)

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

    @Test
    void trainingForPointsInsideACircleShouldConverge() throws Exception {

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 110, 110, 1}, 
            new SimpleBatchGradientDescent(
                new SimpleBatchGradientDescent.HyperParameters(
                        0.1,
                        0.001,
                        1000
                ),
                () -> (Math.random() - 0.5) * 2.0)
        );

        var frame = new JFrame("Thinking");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        var cp = frame.getContentPane();
        frame.setBounds(10, 10, 1500, 1500);
        frame.setVisible(true);
        var g2 = (Graphics2D) cp.getGraphics();
        Thread.sleep(100);
        g2.translate(750, 750);
        g2.setStroke(new BasicStroke(2f));
        g2.setColor(Color.BLUE);
        g2.drawOval(-500, -500, 1000, 1000);

        frame.setTitle("training");

        Random r = new Random(11); 
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .map(i -> {
                double x = r.nextDouble() * 3.0 - 1.5;
                double y = r.nextDouble() * 3.0 - 1.5;
                boolean inside = Math.sqrt(x*x + y*y) <= 1.0;

                if (inside) {
                    g2.setColor(Color.MAGENTA);
                } else {
                    g2.setColor(Color.CYAN);
                }

                g2.fillOval((int) (x*500.0) - 4, (int) (y*500.0) - 4, 8, 8);


                return new TrainingSample(new ConstantVector(new double[]{x, y}), new ConstantVector(new double[]{inside ? 1.0 : 0.0}));
            })
            .collect(Collectors.toList());

        NeuralNetwork network = trainer.train(trainingSet);

        frame.setTitle("testing");
        
        var failures = Stream.iterate(1, i -> i < 100, i -> i+1)
            .flatMap (i -> {
                double x = r.nextDouble() * 3.0 - 1.5;
                double y = r.nextDouble() * 3.0 - 1.5;
                boolean actuallyInside = Math.sqrt(x*x + y*y) <= 1.0;
                boolean predictedInside = Math.round(network.apply(new double[]{x, y})[0]) >= 1;
                if (predictedInside) {
                    g2.setColor(Color.GREEN);
                } else {
                    g2.setColor(Color.RED);
                }
                g2.fillOval((int) (x*500.0) - 4, (int) (y*500.0) - 4, 8, 8);
                if (predictedInside != actuallyInside) {
                    return Stream.of("Bad answer for ("+x+","+y+") distance from origin is " + Math.sqrt(x*x + y*y) + "\n");
                } else {
                    return Stream.empty();
                }
            })
            .collect(Collectors.toList());

            Thread.sleep(100000);

        assertEquals(List.of(), failures, () -> failures.size() + " incorrect predictions");
    }
}
