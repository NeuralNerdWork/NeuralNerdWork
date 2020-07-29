package neuralnerdwork;

import neuralnerdwork.math.Model.ParameterBindings;

@FunctionalInterface
public interface TerminationPredicate {
    boolean shouldContinue(long iterationCount, double[] lastUpdateVector, ParameterBindings currentParameters);
}
