package neuralnerdwork;

import neuralnerdwork.descent.RmsPropUPdate;
import neuralnerdwork.descent.SimpleBatchGradientDescent;
import neuralnerdwork.descent.StochasticGradientDescent;
import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.viz.JFrameTrainingVisualizer;
import org.junit.jupiter.api.Test;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SimpleTrainingTest {
    @Test
    void trainingTwoLayerNetworkShouldConverge() {

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 1}, 
            new SimpleBatchGradientDescent(0.1, () -> (Math.random() - 0.5) * 2.0),
            (iterationCount, network) -> iterationCount < 5000
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
            new int[]{2, 10, 10, 1}, 
            new StochasticGradientDescent(
                200,
                () -> (Math.random() - 0.5) * 2.0,
                () -> new RmsPropUPdate(0.001, 0.9, 1e-8)
        ),
        (iterationCount, network) -> iterationCount < 5000);

        var frame = new JFrame("Thinking");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        var cp = frame.getContentPane();
        frame.setBounds(10, 10, 1000, 1000);
        frame.setVisible(true);
        var g2 = (Graphics2D) cp.getGraphics();
        Thread.sleep(100);
        g2.translate(500, 500);
        g2.setStroke(new BasicStroke(2f));
        g2.setColor(Color.BLUE);
        g2.drawOval(-375, -375, 750, 750);

        frame.setTitle("training");

        Random r = new Random(11); 
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .map(i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                boolean inside = Math.sqrt(x*x + y*y) <= 0.75;

                // if (inside) {
                //     g2.setColor(Color.MAGENTA);
                // } else {
                //     g2.setColor(Color.CYAN);
                // }

                // g2.fillOval((int) (x*500.0) - 4, (int) (y*500.0) - 4, 8, 8);


                return new TrainingSample(new ConstantVector(new double[]{x, y}), new ConstantVector(new double[]{inside ? 1.0 : 0.0}));
            })
            .collect(Collectors.toList());

        NeuralNetwork network = trainer.train(trainingSet);

        frame.setTitle("testing");
        g2.drawOval(-375, -375, 750, 750);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .flatMap (i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                boolean actuallyInside = Math.sqrt(x*x + y*y) <= 0.75;
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

    @Test
    void trainingForPointsInsideACircleShouldConverge2() throws Exception {

        Random r = new Random(11);
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .map(i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                boolean inside = Math.sqrt(x*x + y*y) <= 0.75;
                return new TrainingSample(new ConstantVector(new double[]{x, y}), new ConstantVector(new double[]{inside ? 1.0 : 0.0}));
            })
            .collect(Collectors.toList());

        JFrameTrainingVisualizer visualizer = new JFrameTrainingVisualizer(
                trainingSet,
                new Rectangle2D.Double(-1.0, -1.0, 2.0, 2.0),
                (sample, prediction) -> {
                    boolean predictedInside = prediction.get(0) >= 1.0;
                    //System.out.printf("(%1.3f,%1.3f) inside? %s\n", sample.input().get(0), sample.input().get(1), predictedInside);
                    if (predictedInside) {
                        return Color.GREEN;
                    } else {
                        return Color.RED;
                    }
                });
        // in/out
        visualizer.addShape(new Ellipse2D.Double(-0.75, -0.75, 1.5, 1.5));
        visualizer.addShape(new Line2D.Double(-2.0, 0.0, 2.0, 0.0));
        visualizer.addShape(new Line2D.Double(0.0, -2.0, 0.0, 2.0));

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                new int[]{2, 10, 10, 1},
                new StochasticGradientDescent(
                        200,
                        () -> (Math.random() - 0.5) * 2.0,
                        () -> new RmsPropUPdate(0.001, 0.9, 1e-8)
                ),
                (iterationCount, network) -> iterationCount < 5000,
                visualizer);

        NeuralNetwork network = trainer.train(trainingSet);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .flatMap (i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                boolean actuallyInside = Math.sqrt(x*x + y*y) <= 0.75;
                boolean predictedInside = Math.round(network.apply(new double[]{x, y})[0]) >= 1;
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

    @Test
    void trainingForPointsInsideARingShouldConverge() throws Exception {

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 5, 5, 5, 5, 1}, 
                new StochasticGradientDescent(
                        200,
                        () -> (Math.random() - 0.5) * 2.0,
                        () -> new RmsPropUPdate(0.001, 0.9, 1e-8)
                ),
                (iterationCount, network) -> iterationCount < 5000
        );

        var frame = new JFrame("Thinking");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        var cp = frame.getContentPane();
        frame.setBounds(10, 10, 1000, 1000);
        frame.setVisible(true);
        var g2 = (Graphics2D) cp.getGraphics();
        Thread.sleep(100);
        g2.translate(500, 500);
        g2.setStroke(new BasicStroke(2f));
        g2.setColor(Color.BLUE);
        g2.drawOval(-375, -375, 750, 750);
        g2.drawOval(-125, -125, 250, 250);

        frame.setTitle("training");

        Random r = new Random(11); 
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .map(i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                var distance = Math.sqrt(x*x + y*y);
                boolean inside = distance <= 0.75 && distance >= 0.25 ;

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
        g2.drawOval(-375, -375, 750, 750);
        g2.drawOval(-125, -125, 250, 250);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .flatMap (i -> {
                double x = r.nextDouble() * 2.0 - 1.0;
                double y = r.nextDouble() * 2.0 - 1.0;
                boolean actuallyInside = Math.sqrt(x*x + y*y) <= 0.75;
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
