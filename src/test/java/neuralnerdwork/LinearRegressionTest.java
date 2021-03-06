package neuralnerdwork;

import neuralnerdwork.descent.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static neuralnerdwork.NeuralNetwork.fullyConnectedClassificationNetwork;
import static neuralnerdwork.weight.VariableWeightInitializer.smartRandomWeightInitializer;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LinearRegressionTest {
    private Random rand;

    @BeforeEach
    public void init() {
        rand = new Random(11);
    }

    public static Stream<GradientDescentStrategy> gradientDescentStrategies() {
        Random rand = new Random(33);
        return Stream.of(
                new SimpleBatchGradientDescent(1.0),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new FixedLearningRateGradientUpdate(0.5)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new AverageGradientUpdate(0.5, 5)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new MomentumGradientUpdate(0.1, 0.9)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new NesterovMomentumGradientUpdate(0.1, 0.9)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new AdagradUpdate(0.1, 1e-8)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new AdagradDeltaUpdate(0.9, 1e-4)
                ),
                new StochasticGradientDescent(
                        200,
                        rand,
                        () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
                )
        );
    }

    @ParameterizedTest
    @MethodSource("gradientDescentStrategies")
    void trainingShouldConvergeToLearnCircle(GradientDescentStrategy gradientDescentStrategy) {

        final long total = 1000;
        List<TrainingSample> positiveExamples = Stream.generate(() -> pointInUnitCircle(rand))
                                                      .limit(total / 2)
                                                      .map(p -> new TrainingSample(p.toArray(), p.inRadius(1.0) ? ONE : ZERO))
                                                      .collect(toList());
        List<TrainingSample> negativeExamples = Stream.generate(() -> pointOutOfUnitCircle(rand))
                                                      .limit(total / 2)
                                                      .map(p -> new TrainingSample(p.toArray(), p.inRadius(1.0) ? ONE : ZERO))
                                                      .collect(toList());

        List<TrainingSample> trainingSet = Stream.concat(positiveExamples.stream(), negativeExamples.stream())
                                                 .collect(toList());

        long positiveExampleCount = trainingSet.stream()
                                               .filter(sample -> Arrays.equals(sample.output(), ONE))
                                               .count();

        System.out.println("Positive examples: " + positiveExampleCount);

        int validationSetSize = 1000;
        List<TrainingSample> validationSet = Stream
                .generate(() -> new Point((rand.nextDouble() - 0.5) * 3.0, (rand.nextDouble() - 0.5) * 3.0))
                .limit(validationSetSize)
                .map(p -> new TrainingSample(p.toArray(), p.inRadius(1.0) ? ONE : ZERO))
                .collect(toList());

        record Result(long success, long total) {
            double successRate() {
                return ((double) success) / ((double) total);
            }
        }

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(fullyConnectedClassificationNetwork(smartRandomWeightInitializer(rand), 2, 10, 10, 1), gradientDescentStrategy, (iterationCount, network) -> iterationCount < 5000
                                && validationSet.stream()
                                                .map(sample -> {
                                                    double[] observed = network.apply(sample.input());
                                                    boolean match = Util
                                                            .compareClassifications(observed[0], sample.output()[0]);
                                                    return new Result(match ? 1 : 0, 1);
                                                })
                                                .reduce(new Result(0, 0), (r1, r2) -> new Result(r1.success() + r2
                                                        .success(), r1.total() + r2.total()))
                                                .successRate() < 0.9);

        NeuralNetwork network = trainer.train(trainingSet);

        record Eval(TrainingSample sample, double output) {}

        long trainingSetFailures = trainingSet.stream()
                                              .map(sample -> new Eval(sample, network.apply(sample.input())[0]))
                                              .filter(eval -> Math.round(Math.abs(eval.output() - eval.sample.output()[0])) != 0L)
                                              .count();

        long validationFailures = validationSet.stream()
                .map(sample -> new Eval(sample, network.apply(sample.input())[0]))
                .filter(eval -> Math.round(Math.abs(eval.output() - eval.sample.output()[0])) != 0L)
                .count();

        final double trainingSetAccuracy = 100.0 * (1.0 - trainingSetFailures / (double) trainingSet.size());
        System.out.printf("Training set accuracy: %.2f%%\n", trainingSetAccuracy);
        final double validationSetAccuracy = 100.0 * (1.0 - validationFailures / (double) validationSetSize);
        System.out.printf("Validation set accuracy: %.2f%%\n", validationSetAccuracy);

        assertTrue(trainingSetAccuracy >= 90.0);
        assertTrue(validationSetAccuracy >= 90.0);
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

    private static final double[] ONE = new double[] {1.0};
    private static final double[] ZERO = new double[] {0.0};

    private static record Point(double x, double y) {
        double[] toArray() {
            return new double[] {x, y};
        }
        boolean inRadius(double radius) {
            return Math.sqrt(x*x + y*y) <= radius;
        }
    }
}
