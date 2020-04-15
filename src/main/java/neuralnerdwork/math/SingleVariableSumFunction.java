package neuralnerdwork.math;

public record SingleVariableSumFunction(SingleVariableFunction left, SingleVariableFunction right) implements SingleVariableFunction {

    @Override
    public double apply(double input) {
        return left.apply(input) + right.apply(input);
    }

    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        return new SingleVariableSumFunction(left.differentiateBySingleVariable(), right.differentiateBySingleVariable());
    }
}
