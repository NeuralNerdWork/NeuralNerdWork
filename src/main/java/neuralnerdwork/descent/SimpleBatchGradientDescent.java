package neuralnerdwork.descent;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;
import neuralnerdwork.math.VectorExpression;

import java.util.List;
import java.util.function.Function;

public record SimpleBatchGradientDescent(double trainingRate) implements GradientDescentStrategy {

    @Override
    public Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples,
                                                      Model.ParameterBindings parameterBindings,
                                                      Function<List<TrainingSample>, ScalarExpression> errorFunction,
                                                      TerminationPredicate terminationPredicate) {
        final ScalarExpression error = errorFunction.apply(trainingSamples);
        // use derivative to adjust weights
        final int[] variables = parameterBindings.variables();

        // Repeat this until converged
        Vector weightUpdateVector;
        long iterations = 0;
        do {
            weightUpdateVector = error.computeDerivative(parameterBindings, parameterBindings.variables());
            for (int variableIndex = 0; variableIndex < parameterBindings.variables().length; variableIndex++) {
                int variable = variables[variableIndex];
                parameterBindings.put(variable, parameterBindings.get(variable) - trainingRate * weightUpdateVector.get(variable));
            }
            iterations++;
        } while (terminationPredicate.shouldContinue(iterations, weightUpdateVector, parameterBindings));
        System.out.println("Terminated after " + iterations + " iterations");
        // training cycle end
        // TODO - Stop when we have converged
        return parameterBindings;
    }
}
