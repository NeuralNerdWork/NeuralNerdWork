package neuralnerdwork.math;

public record VectorizedSingleVariableFunctions(SingleVariableFunction... functions) implements VectorFunction {
    public VectorizedSingleVariableFunctions {
        if (functions == null || functions.length == 0) {
            throw new IllegalArgumentException("cannot vectorize null or empty functions");
        }
    }

    @Override
    public int length() {
        return functions.length;
    }

    @Override
    public Vector apply(double[] inputs) {
        if (inputs.length != functions.length) {
            throw new IllegalArgumentException("vectorized function accepts input of length " + functions.length + " but given " + inputs.length);
        }

        final double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = functions[i].apply(inputs[i]);
        }

        return new ConstantVector(outputs);
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        if (variableIndex < 0 || variableIndex >= functions.length) {
            throw new IndexOutOfBoundsException(variableIndex);
        }

        final SingleVariableFunction[] partialDerivatives = new SingleVariableFunction[functions.length];
        for (int i = 0; i < functions.length; i++) {
            if (variableIndex == i) {
                partialDerivatives[i] = functions[i].differentiateBySingleVariable();
            } else {
                partialDerivatives[i] = new ConstantScalar(0.0);
            }
        }

        return new VectorizedSingleVariableFunctions(partialDerivatives);
    }
}
