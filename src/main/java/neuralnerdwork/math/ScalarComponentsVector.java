package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

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
    public boolean columnVector() {
        return true;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        return new DMatrixRMaj(components.length, 1, true,
                               Arrays.stream(components)
                                     .mapToDouble(f -> f.evaluate(bindings))
                                     .toArray()
        );
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new ScalarComponentsVector(
                Arrays.stream(components)
                      .map(f -> new ConstantScalar(f.computePartialDerivative(bindings, variable)))
                      .toArray(ScalarExpression[]::new)
        ).evaluate(bindings);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new TransposeExpression(
                new ColumnMatrix(
                        Arrays.stream(components)
                              .map(f -> f.computeDerivative(bindings))
                              .map(DMatrixColumnVectorExpression::new)
                              .toArray(VectorExpression[]::new)
                )
        ).evaluate(bindings);
    }

    @Override
    public String toString() {
        return "ScalarComponentsVector{" +
                "components=" + Arrays.toString(components) +
                '}';
    }
}
