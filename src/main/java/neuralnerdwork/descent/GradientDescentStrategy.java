package neuralnerdwork.descent;

import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;

import java.util.List;
import java.util.function.Function;

public interface GradientDescentStrategy {
    Model.ParameterBindings runGradientDescent(List<TrainingSample> trainingSamples, Model.ParameterBindings parameterBindings, Function<List<TrainingSample>, ScalarExpression> errorFunction);
}
