package neuralnerdwork.math;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record ScalarMultiplicationFunction(ScalarFunction left,
                                           ScalarFunction right) implements ScalarFunction {
    @Override
    public double apply(VectorVariableBinding input) {
        return left.apply(input) * right.apply(input);
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        final ScalarFunction leftDerivative = left.differentiate(variable);
        final ScalarFunction rightDerivative = right.differentiate(variable);

        // Product rule
        // (fg)' = f'g + fg'
        return new ScalarSumFunction(
                new ScalarMultiplicationFunction(
                        leftDerivative,
                        right
                ),
                new ScalarMultiplicationFunction(
                        left,
                        rightDerivative
                )
        );
    }

    @Override
    public VectorFunction differentiate(VectorVariable variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.concat(
                left.variables().stream(),
                right.variables().stream()
        ).collect(Collectors.toSet());
    }
}
