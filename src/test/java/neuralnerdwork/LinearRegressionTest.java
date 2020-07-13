package neuralnerdwork;

import neuralnerdwork.descent.*;
import neuralnerdwork.math.ConstantVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LinearRegressionTest {
    @Test
    void testBasicLinearRegressionTraining() {
        var r = new Random(1337);

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                new int[]{2, 1},
                (row, col) -> (r.nextDouble() - 0.5) * 2.0,
                new SimpleBatchGradientDescent(0.1),
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

    public static Stream<GradientDescentStrategy> gradientDescentStrategies() {
        return Stream.of(
                new SimpleBatchGradientDescent(1.0),
                new StochasticGradientDescent(
                        200,
                        () -> new FixedLearningRateGradientUpdate(0.5)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new AverageGradientUpdate(0.5, 5)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new MomentumGradientUpdate(0.1, 0.9)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new NesterovMomentumGradientUpdate(0.1, 0.9)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new AdagradUpdate(0.1, 1e-8)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new AdagradDeltaUpdate(0.9, 1e-4)
                ),
                new StochasticGradientDescent(
                        200,
                        () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
                )
        );
    }

    @ParameterizedTest
    @MethodSource("gradientDescentStrategies")
    void trainingShouldConvergeToLearnCircle(GradientDescentStrategy gradientDescentStrategy) {

        final long total = 1000;
        Random r = new Random(11);
        List<TrainingSample> positiveExamples = Stream.generate(() -> pointInUnitCircle(r))
                                                      .limit(total / 2)
                                                      .map(p -> new TrainingSample(p.toVector(), p.inRadius(1.0) ? ONE : ZERO))
                                                      .collect(toList());
        List<TrainingSample> negativeExamples = Stream.generate(() -> pointOutOfUnitCircle(r))
                                                      .limit(total / 2)
                                                      .map(p -> new TrainingSample(p.toVector(), p.inRadius(1.0) ? ONE : ZERO))
                                                      .collect(toList());

        List<TrainingSample> trainingSet = Stream.concat(positiveExamples.stream(), negativeExamples.stream())
                                                 .collect(toList());

        long positiveExampleCount = trainingSet.stream()
                                               .filter(sample -> sample.output().equals(ONE))
                                               .count();

        System.out.println("Positive examples: " + positiveExampleCount);

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(
                new int[]{2, 4, 2, 1},
                (row, col) -> (r.nextDouble() - 0.5) * 2.0,
                gradientDescentStrategy,
                (iterationCount, network) -> iterationCount < 5000
        );

        NeuralNetwork network = trainer.train(trainingSet);

        record Eval(TrainingSample sample, double output) {}

        long trainingSetFailures = trainingSet.stream()
                                              .map(sample -> new Eval(sample, network.apply(sample.input().toArray())[0]))
                                              .filter(eval -> Math.round(Math.abs(eval.output() - eval.sample.output().get(0))) != 0L)
                                              .count();

        int validationSetSize = 1000;
        long validationFailures = Stream.generate(() -> new Point((r.nextDouble() - 0.5) * 3.0, (r.nextDouble() - 0.5) * 3.0))
                                        .limit(validationSetSize)
                                        .map(p -> new TrainingSample(p.toVector(), p.inRadius(1.0) ? ONE : ZERO))
                                        .map(sample -> new Eval(sample, network.apply(sample.input().toArray())[0]))
                                        .filter(eval -> Math.round(Math.abs(eval.output() - eval.sample.output().get(0))) != 0L)
                                        .count();

        final double trainingSetAccuracy = 100.0 * (1.0 - trainingSetFailures / (double) trainingSet.size());
        System.out.printf("Training set accuracy: %.2f%%\n", trainingSetAccuracy);
        final double validationSetAccuracy = 100.0 * (1.0 - validationFailures / (double) validationSetSize);
        System.out.printf("Validation set accuracy: %.2f%%\n", validationSetAccuracy);

        assertTrue(trainingSetAccuracy > 95.0);
        assertTrue(validationSetAccuracy > 90.0);
    }

    private Point pointInUnitCircle(Random random) {
        final double r = random.nextDouble(),
                theta = random.nextDouble() * 2.0 * Math.PI,
                x = r * Math.cos(theta),
                y = r * Math.sin(theta);

        return new Point(x, y);
    }

    private Point pointOutOfUnitCircle(Random random) {
        final double r = 1.0 + random.nextDouble(),
                theta = random.nextDouble() * 2.0 * Math.PI,
                x = r * Math.cos(theta),
                y = r * Math.sin(theta);

        return new Point(x, y);
    }

    private static final ConstantVector ONE = new ConstantVector(new double[] {1.0});
    private static final ConstantVector ZERO = new ConstantVector(new double[] {0.0});

    private static record Point(double x, double y) {
        ConstantVector toVector() {
            return new ConstantVector(new double[] {x, y});
        }
        boolean inRadius(double radius) {
            return Math.sqrt(x*x + y*y) <= radius;
        }
    }
}
