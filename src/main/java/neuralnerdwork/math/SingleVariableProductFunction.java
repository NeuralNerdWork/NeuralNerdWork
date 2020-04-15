package neuralnerdwork.math;

public record SingleVariableProductFunction(SingleVariableFunction left,
                                            SingleVariableFunction right) implements SingleVariableFunction {
    @Override
    public double apply(double input) {
        return left.apply(input) * right.apply(input);
    }

    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        final SingleVariableFunction leftDerivative = left.differentiateBySingleVariable();
        final SingleVariableFunction rightDerivative = right.differentiateBySingleVariable();

        // Product rule
        // (fg)' = f'g + fg'
        return new SingleVariableSumFunction(
                new SingleVariableProductFunction(
                        leftDerivative,
                        right
                ),
                new SingleVariableProductFunction(
                        left,
                        rightDerivative
                )
        );
    }
}
