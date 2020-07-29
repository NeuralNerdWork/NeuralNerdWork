package neuralnerdwork.descent;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

import java.util.List;
import java.util.function.Function;

public record SimpleBatchGradientDescent(double trainingRate) implements GradientDescentStrategy {

    @Override
    public Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples,
                                                      Model.ParameterBindings parameterBindings,
                                                      Function<List<TrainingSample>, ScalarExpression> errorFunction,
                                                      TerminationPredicate terminationPredicate) {
        return new StochasticGradientDescent(trainingSamples
                                                     .size(), () -> new FixedLearningRateGradientUpdate(trainingRate))
                .runGradientDescent(trainingSamples, parameterBindings, errorFunction, terminationPredicate);
    }
}
