package neuralnerdwork.math;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record ScalarSumFunction(ScalarFunction left, ScalarFunction right) implements ScalarFunction {

    @Override
    public double apply(ScalarVariableBinding[] input) {
        return left.apply(input) + right.apply(input);
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        return new ScalarSumFunction(left.differentiate(variable), right.differentiate(variable));
    }

    @Override
    public VectorFunction differentiate(ScalarVariable[] variable) {
        return new VectorSumFunction(left.differentiate(variable), right.differentiate(variable));
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.concat(left.variables()
                                 .stream(),
                             right.variables()
                                  .stream())
                     .collect(Collectors.toSet());
    }
}
