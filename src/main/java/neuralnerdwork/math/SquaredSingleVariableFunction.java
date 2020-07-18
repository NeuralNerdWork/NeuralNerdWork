package neuralnerdwork.math;

public class SquaredSingleVariableFunction implements SingleVariableFunction {
    @Override
    public String getFunctionName() {
        return "squared";
    }

    @Override
    public double apply(double input) {
        return input * input;
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new SingleVariableFunction() {
            @Override
            public String getFunctionName() {
                return "squared derivative";
            }

            @Override
            public double apply(double input) {
                return 2.0 * input;
            }

            @Override
            public SingleVariableFunction differentiateByInput() {
                throw new UnsupportedOperationException("Not implemented");
            }
        };
    }
}
