package neuralnerdwork.descent;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

public record StochasticGradientDescent(int batchSize,
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
        final Random rand = new Random();
        final WeightUpdateStrategy updateStrategy = updateStrategySupplier.get();
        long iterations = 0;
        do {
            Collections.shuffle(trainingSamples, rand);
            final int batchSize = Math.min(batchSize(), trainingSamples.size());
            final List<TrainingSample> iterationSamples = trainingSamples.subList(0, batchSize);
            final ScalarExpression error = errorFunction.apply(iterationSamples);
            // use derivative to adjust weights
            weightUpdateVector = updateStrategy.updateVector(error, parameterBindings);
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
