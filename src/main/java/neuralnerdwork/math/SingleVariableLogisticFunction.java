package neuralnerdwork.math;

public class SingleVariableLogisticFunction implements SingleVariableFunction {

    @Override
    public double apply(double input) {
        final double eToX = Math.exp(input);
        return eToX / (1 + eToX);
    }

    @Override
    public SingleVariableFunction differentiateBySingleVariable() {
        return new SingleVariableProductFunction(this, new SingleVariableComposition(new NegateScalar(), this));
    }
}
