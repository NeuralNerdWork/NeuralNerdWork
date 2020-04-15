package neuralnerdwork.math;

public interface ScalarFunction {
    int inputLength();
    double apply(double[] inputs);
    ScalarFunction differentiate(int variableIndex);
    default VectorFunction differentiate() {
        final ScalarFunction[] partialDerivatives = new ScalarFunction[inputLength()];
        for (int i = 0; i < inputLength(); i++) {
            partialDerivatives[i] = differentiate(i);
        }

        return new ScalarComponentsFunction(partialDerivatives);
    }
}
