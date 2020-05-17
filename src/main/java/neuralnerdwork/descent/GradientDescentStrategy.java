package neuralnerdwork.descent;

import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.VectorExpression;

import java.util.List;
import java.util.function.Function;

public interface GradientDescentStrategy {
    Model.Binder runGradientDescent(List<TrainingSample> trainingSamples, Model.Binder binder, Function<List<TrainingSample>, ScalarExpression> errorFunction);
}
