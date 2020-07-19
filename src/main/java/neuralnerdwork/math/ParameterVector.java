package neuralnerdwork.math;

import java.util.Arrays;
import java.util.stream.IntStream;

public record ParameterVector(int variableStartIndex, int length) implements VectorExpression {

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = bindings.get(variableStartIndex + i);
        }

        return new ConstantVector(values);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        if (variable >= variableStartIndex && variable < variableStartIndex + length) {
            final double[] values = new double[length];
            values[variable - variableStartIndex] = 1.0;

            return new ConstantVector(values);
        } else {
            return new ConstantVector(new double[length]);
        }
    }

    @Override
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        return new ColumnMatrix(
                Arrays.stream(variables)
                      .mapToObj(variable -> computePartialDerivative(bindings, variable))
                      .toArray(VectorExpression[]::new)
        ).evaluate(bindings);
    }

    public boolean containsVariable(int variable) {
        return variable >= variableStartIndex && variable < (variableStartIndex + length);
    }

    @Override
    public boolean isZero() {
        return false;
    }

    public int indexFor(int variable) {
        return variable - variableStartIndex;
    }

    public int variableFor(int index) {
        return variableStartIndex + index;
    }

    public IntStream variables() {
        return IntStream.iterate(variableStartIndex, n -> n + 1)
                        .limit(length);
    }
}
