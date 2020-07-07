package neuralnerdwork;

import neuralnerdwork.descent.RmsPropUpdate;
import neuralnerdwork.descent.SimpleBatchGradientDescent;
import neuralnerdwork.descent.StochasticGradientDescent;
import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.viz.JFrameTrainingVisualizer;
import org.junit.jupiter.api.Test;

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
                    boolean predictedInside = prediction.get(0) >= 0.5;
                    //System.out.printf("(%1.3f,%1.3f) inside? %s\n", sample.input().get(0), sample.input().get(1), predictedInside);
                    if (predictedInside) {
                        return Color.GREEN;
                    } else {
                        return Color.RED;
                    }
                });

        visualizer.addShape(new Ellipse2D.Double(-0.75, -0.75, 1.5, 1.5));
        visualizer.addShape(new Line2D.Double(-2.0, 0.0, 2.0, 0.0));
        visualizer.addShape(new Line2D.Double(0.0, -2.0, 0.0, 2.0));

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                new int[]{2, 10, 10, 1},
                new StochasticGradientDescent(
                        200,
                        () -> (Math.random() - 0.5) * 2.0,
                        () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
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

        System.in.read();
        assertEquals(List.of(), failures, () -> failures.size() + " incorrect predictions");
    }

    @Test
    void trainingForPointsInsideARingShouldConverge() throws Exception {

        Random r = new Random(11); 
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i+1)
            .map(i -> {
                double x = r.nextGaussian() * 0.5;
                double y = r.nextGaussian() * 0.5;
                var distance = Math.sqrt(x*x + y*y);
                boolean inside = distance <= 0.75 && distance >= 0.25;

                return new TrainingSample(new ConstantVector(new double[]{x, y}), new ConstantVector(new double[]{inside ? 1.0 : 0.0}));
            })
            .collect(Collectors.toList());

        JFrameTrainingVisualizer visualizer = new JFrameTrainingVisualizer(
            trainingSet,
            new Rectangle2D.Double(-2.0, -2.0, 4.0, 4.0),
            (sample, prediction) -> {
                // System.out.printf("(%1.3f,%1.3f) inside? %1.3f\n", sample.input().get(0), sample.input().get(1), prediction.get(0));
                var greenAmt = (int) Math.round((prediction.get(0)-0.5)* 2.0 * 255);
                var redAmt = (int) Math.round((0.5 - prediction.get(0))* 2.0 * 255);

                if(prediction.get(0) >= 0.5) {
                    return new Color(0, greenAmt, 0);
                } else {
                    return new Color(redAmt, 0, 0);
                }
            });
        // in/out
        visualizer.addShape(new Ellipse2D.Double(-0.75, -0.75, 1.5, 1.5));
        visualizer.addShape(new Ellipse2D.Double(-0.25, -0.25, 0.5, 0.5));
        visualizer.addShape(new Line2D.Double(-2.0, 0.0, 2.0, 0.0));
        visualizer.addShape(new Line2D.Double(0.0, -2.0, 0.0, 2.0));

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
            new int[]{2, 10, 10, 1}, 
                new StochasticGradientDescent(
                        100,
                        () -> (Math.random() - 0.5) * 2,
                        () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
                ),
                (iterationCount, network) -> iterationCount < 5000,
                visualizer
        );


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

        System.in.read();
        assertEquals(List.of(), failures, () -> failures.size() + " incorrect predictions");
    }

}
