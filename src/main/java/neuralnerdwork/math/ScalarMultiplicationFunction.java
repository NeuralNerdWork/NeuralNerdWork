package neuralnerdwork.math;

public record ScalarMultiplicationFunction(ScalarFunction left,
                                           ScalarFunction right) implements ScalarFunction {
    @Override
    public double apply(double[] input) {
        return left.apply(input) * right.apply(input);
    }

    @Override
    public ScalarFunction differentiate(int variableIndex) {
        final ScalarFunction leftDerivative = left.differentiate(variableIndex);
        final ScalarFunction rightDerivative = right.differentiate(variableIndex);

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
}
