package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarComponentsVector(ScalarExpression[] components) implements VectorExpression {
    public ScalarComponentsVector {
        if (components == null || components.length == 0) {
            throw new IllegalArgumentException("Cannot have null or empty components");
        }
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(components)
                     .allMatch(ScalarExpression::isZero);
    }

    @Override
    public int length() {
        return components.length;
    }

    @Override
    public Vector evaluate(Model.Binder bindings) {
        return new ConstantVector(
                Arrays.stream(components)
                      .mapToDouble(f -> f.evaluate(bindings))
                      .toArray()
        );
    }

    @Override
    public VectorExpression computePartialDerivative(int variable) {
        return new ScalarComponentsVector(
                Arrays.stream(components)
                      .map(f -> f.computePartialDerivative(variable))
                      .toArray(ScalarExpression[]::new)
        );
    }

    @Override
    public MatrixExpression computeDerivative(int[] variables) {
        return new TransposeExpression(
                new ColumnMatrix(
                        Arrays.stream(components)
                              .map(f -> f.computeDerivative(variables))
                              .toArray(VectorExpression[]::new)
                )
        );
    }

    @Override
    public String toString() {
        return "ScalarComponentsVector{" +
                "components=" + Arrays.toString(components) +
                '}';
    }
}
