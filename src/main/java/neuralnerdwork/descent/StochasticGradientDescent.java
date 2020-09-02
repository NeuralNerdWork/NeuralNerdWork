package neuralnerdwork.descent;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

public record StochasticGradientDescent(int batchSize,
                                        Random rand,
                                        Supplier<WeightUpdateStrategy> updateStrategySupplier) implements GradientDescentStrategy {

    @Override
    public Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples,
                                                      Model.ParameterBindings parameterBindings,
                                                      Function<List<TrainingSample>, ScalarExpression> errorFunction,
                                                      TerminationPredicate terminationPredicate) {
        // Make a copy that we can shuffle
        trainingSamples = new ArrayList<>(trainingSamples);

        // Repeat this until converged
        double[] weightUpdateVector;
        final WeightUpdateStrategy updateStrategy = updateStrategySupplier.get();
        long iterations = 0;
        do {
            Collections.shuffle(trainingSamples, rand);
            final int batchSize = Math.min(batchSize(), trainingSamples.size());
            final List<TrainingSample> iterationSamples = trainingSamples.subList(0, batchSize);
            var start = Instant.now();
            final ScalarExpression error = errorFunction.apply(iterationSamples);
            System.out.printf("Time for error function construction: %dms\n", java.time.Duration.between(start, Instant.now()).toMillis());
            // use derivative to adjust weights
            start = Instant.now();
            weightUpdateVector = updateStrategy.updateVector(error, parameterBindings);
            System.out.printf("Time for update vector evaluation: %dms\n", java.time.Duration.between(start, Instant.now()).toMillis());
            for (int variable : parameterBindings.variables()) {
                parameterBindings.put(variable, parameterBindings.get(variable) + weightUpdateVector[variable]);
            }
            iterations++;
        } while (terminationPredicate.shouldContinue(iterations, weightUpdateVector, parameterBindings));
        System.out.println("Terminated after " + iterations + " iterations");
        // training cycle end
        // TODO - Stop when we have converged
        return parameterBindings;
    }
}
