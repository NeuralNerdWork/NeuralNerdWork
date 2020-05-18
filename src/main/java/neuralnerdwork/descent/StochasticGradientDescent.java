package neuralnerdwork.descent;

import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.function.Supplier;

public record StochasticGradientDescent(HyperParameters hyperParameters,
                                        DoubleSupplier initialWeightSupplier,
                                        Supplier<WeightUpdateStrategy> updateStrategySupplier) implements GradientDescentStrategy {
    public static record HyperParameters(double convergenceThreshold, long maxIterations, int batchSize) {}

    @Override
    public Model.Binder runGradientDescent(List<TrainingSample> trainingSamples,
                                           Model.Binder binder,
                                           Function<List<TrainingSample>, ScalarExpression> errorFunction) {
        // Make a copy that we can shuffle
        trainingSamples = new ArrayList<>(trainingSamples);
        final int[] variables = binder.variables();
        // initialize weights
        for (int variable : variables) {
            binder.put(variable, initialWeightSupplier.getAsDouble());
        }

        // Repeat this until converged
        Vector weightUpdateVector;
        final Random rand = new Random();
        final WeightUpdateStrategy updateStrategy = updateStrategySupplier.get();
        long iterations = 0;
        do {
            Collections.shuffle(trainingSamples, rand);
            final List<TrainingSample> iterationSamples = trainingSamples.subList(0, hyperParameters.batchSize());
            final ScalarExpression error = errorFunction.apply(iterationSamples);
            // use derivative to adjust weights
            final Vector rawGradient = error.computeDerivative(binder.variables())
                                         .evaluate(binder);
            weightUpdateVector = updateStrategy.updateVector(rawGradient);
            for (int variableIndex = 0; variableIndex < binder.variables().length; variableIndex++) {
                int variable = variables[variableIndex];
                binder.put(variable, binder.get(variable) + weightUpdateVector.get(variable));
            }
            if (iterations % 10 == 0) {
                System.out.println("Completed iteration " + iterations);
                System.out.println("  update vector: " + weightUpdateVector);
                System.out.println("  update vector length: " + weightUpdateVector.lTwoNorm());
            }
            iterations++;
        } while (weightUpdateVector.lTwoNorm() > hyperParameters.convergenceThreshold() && iterations < hyperParameters.maxIterations());
        System.out.println("Terminated after " + iterations + " iterations");
        // training cycle end
        // TODO - Stop when we have converged
        return binder;
    }
}
