package neuralnerdwork;

import neuralnerdwork.math.Model.ParameterBindings;
import neuralnerdwork.math.Vector;

@FunctionalInterface
public interface TerminationPredicate {
    boolean shouldContinue(long iterationCount, Vector lastUpdateVector, ParameterBindings currentParameters);
}
