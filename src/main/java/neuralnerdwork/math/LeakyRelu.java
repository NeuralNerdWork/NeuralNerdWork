package neuralnerdwork.math;

public class LeakyRelu implements SingleVariableFunction {

    private final double alpha;

    public LeakyRelu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public String getFunctionName() {
        return "relu";
    }

    @Override
    public double apply(double input) {
        if (input >= 0.0) {
            return input;
        } else {
            return input * alpha;
        }
    }

    @Override
    public SingleVariableFunction differentiateByInput() {
        return new ReluDerivative();
    }

    private class ReluDerivative implements SingleVariableFunction {
        @Override
        public String getFunctionName() {
            return "relu derivative";
        }

        @Override
        public double apply(double input) {
            if (input < 0) {
                return alpha;
            } else {
                return 1;
            }
        }

        @Override
        public SingleVariableFunction differentiateByInput() {
            throw new UnsupportedOperationException("Not yet implemented!");
        }
    }
}
