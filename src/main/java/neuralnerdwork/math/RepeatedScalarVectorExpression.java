package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record RepeatedScalarVectorExpression(ScalarExpression scalar, int length) implements VectorExpression {

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        return new RepeatedScalarVector(scalar.evaluate(bindings), length);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new RepeatedScalarVector(scalar.computePartialDerivative(bindings, variable), length);
    }

    @Override
    public boolean isZero() {
        return scalar.isZero();
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        double[] gradient = scalar.computeDerivative(bindings).toArray();
        double[] values = new double[length * gradient.length];
        for (int i = 0; i < values.length; i++) {
            System.arraycopy(gradient, 0, values, gradient.length * i, gradient.length);
        }

        return new DMatrixRMaj(length, gradient.length, true, values);
    }
}
