package neuralnerdwork.math;

import java.util.Arrays;

public record ParameterVector(int variableStartIndex, int length) implements VectorExpression {

    @Override
    public Vector evaluate(Model.Binder bindings) {
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = bindings.get(variableStartIndex + i);
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        if (variable >= variableStartIndex && variable < variableStartIndex + length) {
            final double[] values = new double[length];
            values[variable - variableStartIndex] = 1.0;

            return new ConstantVector(values);
        } else {
            return new ConstantVector(new double[length]);
        }
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return new ColumnMatrix(
                Arrays.stream(variables)
                      .mapToObj(this::computePartialDerivative)
                      .toArray(VectorExpression[]::new)
        );
    }

    @Override
    public boolean isZero() {
        return false;
    }
}
