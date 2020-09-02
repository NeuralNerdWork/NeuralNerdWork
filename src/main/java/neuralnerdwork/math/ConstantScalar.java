package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

public record ConstantScalar(double value) implements ScalarExpression {
    public static final ConstantScalar ZERO = new ConstantScalar(0.0);

    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return value;
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return 0.0;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new DMatrixSparseCSC(1, bindings.size(), 0);
    }

    @Override
    public boolean isZero() {
        return value == 0.0;
    }
}
