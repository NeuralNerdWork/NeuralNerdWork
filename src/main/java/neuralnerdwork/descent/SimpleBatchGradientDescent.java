package neuralnerdwork.descent;

import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;
import neuralnerdwork.math.VectorExpression;

import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.Function;

public record SimpleBatchGradientDescent(HyperParameters hyperParameters, DoubleSupplier initialWeightSupplier) implements GradientDescentStrategy {
    public static record HyperParameters(double trainingRate, double convergenceThreshold, long maxIterations) {}

    @Override
    public Model.Binder runGradientDescent(List<TrainingSample> trainingSamples,
                                           Model.Binder binder,
                                           Function<List<TrainingSample>, ScalarExpression> errorFunction) {
        final ScalarExpression error = errorFunction.apply(trainingSamples);
        // use derivative to adjust weights
        VectorExpression lossDerivative = error.computeDerivative(binder.variables());
        final int[] variables = binder.variables();
        // initialize weights
        for (int variable : variables) {
            binder.put(variable, initialWeightSupplier.getAsDouble());
        }

        // Repeat this until converged
        Vector weightUpdateVector;
        long iterations = 0;
        do {
            weightUpdateVector = lossDerivative.evaluate(binder);
            for (int variableIndex = 0; variableIndex < binder.variables().length; variableIndex++) {
                int variable = variables[variableIndex];
                binder.put(variable, binder.get(variable) - hyperParameters.trainingRate() * weightUpdateVector.get(variable));
            }
            if (iterations % 10 == 0) {
                System.out.println("Completed iteration " + iterations);
                System.out.println("  gradient: " + weightUpdateVector);
                System.out.println("  gradient length: " + weightUpdateVector.lTwoNorm());
            }
            iterations++;
        } while (weightUpdateVector.lTwoNorm() > hyperParameters.convergenceThreshold() && iterations < hyperParameters.maxIterations());
        System.out.println("Terminated after " + iterations + " iterations");
        // training cycle end
        // TODO - Stop when we have converged
        return binder;
    }
}
