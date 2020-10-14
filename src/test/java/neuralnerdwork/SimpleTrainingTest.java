package neuralnerdwork;

import static neuralnerdwork.NeuralNetwork.fullyConnectedClassificationNetwork;
import static neuralnerdwork.weight.VariableWeightInitializer.smartRandomWeightInitializer;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.math.BigInteger;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

import neuralnerdwork.descent.RmsPropUpdate;
import neuralnerdwork.descent.SimpleBatchGradientDescent;
import neuralnerdwork.descent.StochasticGradientDescent;
import neuralnerdwork.viz.JFrameTrainingVisualizer;

public class SimpleTrainingTest {
    // TODO - attempt word2vec sized problem?
    // TODO - Visualize weights during training (maybe compare regularized vs not)
    // TODO - Drop out
    // TODO - Do multiple classifications with softmax
    // TODO - Add check in NeuralNetwork.apply for input vector sizes
    // TODO - investigate training visualizations that work in >2D

    /*
     Perf TODOs
     - Check out self-time of computeDerivative
     - Refactor compute derivative so forward, backwards, and last part are in own methods (for profiling)
     - Maybe refactor backwards and last parts to single part? (Might not make a difference)
     */

    static record FailurePercent(int failures, int total) {
        FailurePercent merge(FailurePercent other) {
            return new FailurePercent(failures() + other.failures(), total() + other.total());
        }

        double asPercent() {
            return ((double) failures) / ((double) total);
        }

    }

    @Test
    void trainingTwoLayerNetworkShouldConverge() {

        Random r = new Random(11);
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                fullyConnectedClassificationNetwork(smartRandomWeightInitializer(r), 2, 1),
                new SimpleBatchGradientDescent(1.0),
                (iterationCount, network) -> network.apply(new double[] {0.0, 0.1})[0] > 0.2
                        || network.apply(new double[] {0.0, 1.3})[0] < 0.8
        );

        NeuralNetwork network = trainer.train(Arrays.asList(
                new TrainingSample(new double[] {0.0, 0.1}, new double[] {0.0}),
                new TrainingSample(new double[] {0.0, 1.3}, new double[] {1.0})
        ));

        assertArrayEquals(new double[] {0.0}, network.apply(new double[] {0.0, 0.1}), 0.2);
        assertArrayEquals(new double[] {1.0}, network.apply(new double[] {0.0, 1.3}), 0.2);
    }

    @Test
    void trainingForPointsInsideACircleShouldConverge() throws Exception {

        Random r = new Random(11);
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    boolean inside = Math.sqrt(x * x + y * y) <= 0.75;
                    return new TrainingSample(new double[] {x, y}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        var verificationSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    boolean inside = Math.sqrt(x * x + y * y) <= 0.75;
                    return new TrainingSample(new double[] {x, y}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        JFrameTrainingVisualizer visualizer = new JFrameTrainingVisualizer(
                trainingSet,
                new Rectangle2D.Double(-1.0, -1.0, 2.0, 2.0),
                (sample, prediction) -> {
                    boolean predictedInside = prediction[0] >= 0.5;
                    if (predictedInside) {
                        return Color.GREEN;
                    } else {
                        return Color.RED;
                    }
                });

        visualizer.addShape(new Ellipse2D.Double(-0.75, -0.75, 1.5, 1.5));
        visualizer.addShape(new Line2D.Double(-2.0, 0.0, 2.0, 0.0));
        visualizer.addShape(new Line2D.Double(0.0, -2.0, 0.0, 2.0));

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(fullyConnectedClassificationNetwork(
                smartRandomWeightInitializer(r),
                2,
                10,
                10,
                1), new StochasticGradientDescent(
                200,
                r,
                () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
        ), (iterationCount, network) -> {
            var fails = verificationSet.stream()
                    .map(i -> {
                        return Util.compareClassifications(network.apply(i.input())[0], i.output()[0]);
                    })
                    .map(b -> new FailurePercent(b ? 0 : 1, 1))
                    .reduce(new FailurePercent(0, 0), FailurePercent::merge);

            System.out.println("Percentage of verification set failing: " + fails.asPercent());

            return fails.asPercent() > 0.05;
        }, visualizer, NeuralNetworkTrainer.L2NormAdditionalError(0.001));

        NeuralNetwork network = trainer.train(trainingSet);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .flatMap(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    boolean actuallyInside = Math.sqrt(x * x + y * y) <= 0.75;
                    boolean predictedInside = Math.round(network.apply(new double[] {x, y})[0]) >= 1;
                    if (predictedInside != actuallyInside) {
                        return Stream.of("Bad answer for (" + x + "," + y + ") distance from origin is " + Math.sqrt(
                                x * x + y * y) + "\n");
                    } else {
                        return Stream.empty();
                    }
                })
                .collect(Collectors.toList());

        assertTrue(failures.size() <= 100, () -> failures.size() + " incorrect predictions");
    }

    @Test
    void trainingForPointsInsideARingShouldConverge() throws Exception {
        var start = Instant.now();
        Random r = new Random(12);
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextGaussian() * 0.5;
                    double y = r.nextGaussian() * 0.5;
                    var distance = Math.sqrt(x * x + y * y);
                    boolean inside = distance <= 0.75 && distance >= 0.25;

                    return new TrainingSample(new double[] {x, y}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        var verificationSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextGaussian() * 0.5;
                    double y = r.nextGaussian() * 0.5;
                    var distance = Math.sqrt(x * x + y * y);
                    boolean inside = distance <= 0.75 && distance >= 0.25;

                    return new TrainingSample(new double[] {x, y}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        JFrameTrainingVisualizer visualizer = new JFrameTrainingVisualizer(
                trainingSet,
                new Rectangle2D.Double(-2.0, -2.0, 4.0, 4.0),
                (sample, prediction) -> {
                    // System.out.printf("(%1.3f,%1.3f) inside? %1.3f\n", sample.input().get(0), sample.input().get(1), prediction.get(0));
                    var greenAmt = (int) Math.floor((prediction[0] - 0.5) * 2.0 * 127) + 127;
                    var redAmt = (int) Math.floor((0.5 - prediction[0]) * 2.0 * 127) + 127;

                    if (prediction[0] >= 0.5) {
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

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(fullyConnectedClassificationNetwork(
                smartRandomWeightInitializer(r),
                2,
                20,
                20,
                1), new StochasticGradientDescent(
                100,
                r,
                () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
        ), (iterationCount, network) -> {
            var fails = verificationSet.stream()
                    .map(i -> {
                        return Util.compareClassifications(network.apply(i.input())[0], i.output()[0]);
                    })
                    .map(b -> new FailurePercent(b ? 0 : 1, 1))
                    .reduce(new FailurePercent(0, 0), FailurePercent::merge);

            System.out.println("Percentage of verification set failing: " + fails.asPercent());

            return fails.asPercent() > 0.05;
        }, visualizer, NeuralNetworkTrainer.L2NormAdditionalError(0.001));

        NeuralNetwork network = trainer.train(trainingSet);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .flatMap(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    boolean actuallyInside = Math.sqrt(x * x + y * y) <= 0.75;
                    boolean predictedInside = Math.round(network.apply(new double[] {x, y})[0]) >= 1;

                    if (predictedInside != actuallyInside) {
                        return Stream.of("Bad answer for (" + x + "," + y + ") distance from origin is " + Math.sqrt(
                                x * x + y * y) + "\n");
                    } else {
                        return Stream.empty();
                    }
                })
                .collect(Collectors.toList());

        var end = Instant.now();
        System.out.printf("Total test time %d", Duration.between(start, end).toMillis());
        assertTrue(failures.size() <= 100, () -> failures.size() + " incorrect predictions");
    }

    @Test
    void trainingForPointsInsideASphereShouldConverge() throws Exception {

        Random r = new Random(11);
        var trainingSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    double z = r.nextDouble() * 2.0 - 1.0;
                    boolean inside = Math.sqrt(x * x + y * y + z * z) <= 0.75;
                    return new TrainingSample(new double[] {x, y, z}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        var verificationSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    double z = r.nextDouble() * 2.0 - 1.0;
                    boolean inside = Math.sqrt(x * x + y * y + z * z) <= 0.75;
                    return new TrainingSample(new double[] {x, y, z}, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        JFrameTrainingVisualizer visualizer = new JFrameTrainingVisualizer(
                trainingSet,
                new Rectangle2D.Double(-1.0, -1.0, 2.0, 2.0),
                (sample, prediction) -> {
                    boolean predictedInside = prediction[0] >= 0.5;
                    if (predictedInside) {
                        return Color.GREEN;
                    } else {
                        return Color.RED;
                    }
                });

        visualizer.addShape(new Ellipse2D.Double(-0.75, -0.75, 1.5, 1.5));
        visualizer.addShape(new Line2D.Double(-2.0, 0.0, 2.0, 0.0));
        visualizer.addShape(new Line2D.Double(0.0, -2.0, 0.0, 2.0));

        NeuralNetworkTrainer trainer =
                new NeuralNetworkTrainer(
                    fullyConnectedClassificationNetwork(smartRandomWeightInitializer(r), 3, 10, 10, 1),
                    new StochasticGradientDescent(200, r,
                        () -> new RmsPropUpdate(0.001, 0.9, 1e-8)),
                    (iterationCount, network) -> {
            var fails = verificationSet.stream()
                    .map(i -> {
                        return Util.compareClassifications(network.apply(i.input())[0], i.output()[0]);
                    })
                    .map(b -> new FailurePercent(b ? 0 : 1, 1))
                    .reduce(new FailurePercent(0, 0), FailurePercent::merge);

            System.out.println("Percentage of verification set failing: " + fails.asPercent());

            return fails.asPercent() > 0.05;
        }, visualizer, NeuralNetworkTrainer.L2NormAdditionalError(0.001));

        NeuralNetwork network = trainer.train(trainingSet);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .flatMap(i -> {
                    double x = r.nextDouble() * 2.0 - 1.0;
                    double y = r.nextDouble() * 2.0 - 1.0;
                    double z = r.nextDouble() * 2.0 - 1.0;
                    boolean actuallyInside = Math.sqrt(x * x + y * y + z * z) <= 0.75;
                    boolean predictedInside = Math.round(network.apply(new double[] {x, y, z})[0]) >= 1;
                    if (predictedInside != actuallyInside) {
                        return Stream.of("Bad answer for (" + x + "," + y + "," + z + ") distance from origin is " +
                                                 Math.sqrt(x * x + y * y + z * z) + "\n");
                    } else {
                        return Stream.empty();
                    }
                })
                .collect(Collectors.toList());

        assertTrue(failures.size() <= 100, () -> failures.size() + " incorrect predictions");
    }


    @Test
    void trainingForPointsInsideAnNSphereShouldConverge() throws Exception {

        Random r = new Random(11);
        int dimensions = 10000;
        var volume = Math.pow(2, dimensions) / 2.0;
        var radius = 1.6;
//                Math.pow(Math.PI * dimensions, 1.0 / (2.0 * dimensions)) * Math.sqrt(dimensions / (2.0 * Math.PI * Math.E)) * Math.pow(volume, 1.0 / dimensions);
        var positiveTrainingSet = Stream.generate(() -> 0)
                .map(i -> {
                    var inputs = new double[dimensions];
                    var distanceSum = 0.0;
                    for(int d = 0; d < dimensions; d++) {
                        inputs[d] = r.nextDouble() * 2.0 - 1.0;
                        distanceSum += inputs[d] * inputs[d];
                    }

                    double distance = Math.sqrt(distanceSum);
                    boolean inside = distance <= radius;
//                    System.out.println(i + " : "+ Math.sqrt(distanceSum) + " : " + Arrays.toString(inputs));

                    return new TrainingSample(inputs, new double[] {inside ? 1.0 : 0.0});
                })
//                .filter(ts -> ts.output()[0] > 0.5)
                .limit(500)
                .collect(Collectors.toList());

        var negativeTrainingSet = Stream.generate(() -> 0)
                .map(i -> {
                    var inputs = new double[dimensions];
                    var distanceSum = 0.0;
                    for(int d = 0; d < dimensions; d++) {
                        inputs[d] = r.nextDouble() * 2.0 - 1.0;
                        distanceSum += inputs[d] * inputs[d];
                    }

                    boolean inside = Math.sqrt(distanceSum) <= radius;
                    //                    System.out.println(i + " : "+ Math.sqrt(distanceSum) + " : " + Arrays.toString(inputs));

                    return new TrainingSample(inputs, new double[] {inside ? 1.0 : 0.0});
                })
//                .filter(ts -> ts.output()[0] < 0.5)
                .limit(500)
                .collect(Collectors.toList());

        var trainingSet = new ArrayList<TrainingSample>();
        trainingSet.addAll(positiveTrainingSet);
        trainingSet.addAll(negativeTrainingSet);

        long positiveExamples = trainingSet.stream()
                .filter(sample -> sample.output()[0] == 1.0)
                .count();


        System.out.println("Positive examples: " + positiveExamples + " less than " + radius + " away");

        var verificationSet = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .map(i -> {
                    var inputs = new double[dimensions];
                    var distanceSum = 0.0;
                    for(int d = 0; d < dimensions; d++) {
                        inputs[d] = r.nextDouble() * 2.0 - 1.0;
                        distanceSum += inputs[d] * inputs[d];
                    }

                    boolean inside = Math.sqrt(distanceSum) <= radius;
                    return new TrainingSample(inputs, new double[] {inside ? 1.0 : 0.0});
                })
                .collect(Collectors.toList());

        NeuralNetworkTrainer trainer =
                new NeuralNetworkTrainer(
                        fullyConnectedClassificationNetwork(smartRandomWeightInitializer(r), dimensions, 1000, dimensions, 1),
                        new StochasticGradientDescent(200, r,
                                                      () -> new RmsPropUpdate(0.001, 0.9, 1e-8)),
                        (iterationCount, network) -> {
                            var fails = verificationSet.stream()
                                    .map(i -> {
                                        return Util.compareClassifications(network.apply(i.input())[0], i.output()[0]);
                                    })
                                    .map(b -> new FailurePercent(b ? 0 : 1, 1))
                                    .reduce(new FailurePercent(0, 0), FailurePercent::merge);

                            System.out.println("Percentage of verification set failing: " + fails.asPercent());

                            return iterationCount <= 2 && fails.asPercent() > 0.05;
                        }, (a, b) -> {}, NeuralNetworkTrainer.L2NormAdditionalError(0.001));

        NeuralNetwork network = trainer.train(trainingSet);

        var failures = Stream.iterate(1, i -> i < 1000, i -> i + 1)
                .flatMap(i -> {
                    var inputs = new double[dimensions];
                    var distanceSum = 0.0;
                    for(int d = 0; d < dimensions; d++) {
                        inputs[d] = r.nextDouble() * 2.0 - 1.0;
                        distanceSum += inputs[d] * inputs[d];
                    }

                    double distance = Math.sqrt(distanceSum);
                    boolean actuallyInside = distance <= radius;
                    boolean predictedInside = Math.round(network.apply(inputs)[0]) >= 1;
                    if (predictedInside != actuallyInside) {
                        return Stream.of("Bad answer for (" + Arrays.toString(inputs) + ") distance from origin is " +
                                                 distance + "\n");
                    } else {
                        return Stream.empty();
                    }
                })
                .collect(Collectors.toList());

        assertTrue(failures.size() <= 100, () -> failures.size() + " incorrect predictions");
    }

}
