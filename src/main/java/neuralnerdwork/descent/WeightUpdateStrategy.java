package neuralnerdwork.descent;

import neuralnerdwork.math.Model;
import neuralnerdwork.math.ScalarExpression;
import neuralnerdwork.math.Vector;

public interface WeightUpdateStrategy {
    Vector updateVector(ScalarExpression error, Model.ParameterBindings parameterBindings);
}
