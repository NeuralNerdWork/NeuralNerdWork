package neuralnerdwork.math;

public interface SingleVariableFunction extends ScalarFunction {
    @Override
    default double apply(double[] inputs) {
        if (inputs.length != 1) {
            throw new IllegalArgumentException("only applicable for single variable");
        }

        return apply(inputs[0]);
    }

    @Override
    default int inputLength() {
        return 1;
    }

    @Override
    default ScalarFunction differentiate(int variableIndex) {
        if (variableIndex != 1) {
            throw new IllegalArgumentException("only applicable for single variable");
        }

        return differentiateBySingleVariable();
    }

    double apply(double input);
    SingleVariableFunction differentiateBySingleVariable();
}
