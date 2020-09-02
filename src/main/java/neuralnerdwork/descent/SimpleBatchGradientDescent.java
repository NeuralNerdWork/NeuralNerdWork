package neuralnerdwork.descent;

import java.util.List;
import java.util.Random;
import java.util.function.Function;

import neuralnerdwork.TerminationPredicate;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

public record SimpleBatchGradientDescent(double trainingRate) implements GradientDescentStrategy {

    @Override
    public Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples,
            Model.ParameterBindings parameterBindings,
            Function<List<TrainingSample>, ScalarExpression> errorFunction,
            TerminationPredicate terminationPredicate) {
        return new StochasticGradientDescent(trainingSamples.size(),
                                             new Random(),
                                             () -> new FixedLearningRateGradientUpdate(trainingRate))
                .runGradientDescent(trainingSamples, parameterBindings, errorFunction, terminationPredicate);
    }
}
