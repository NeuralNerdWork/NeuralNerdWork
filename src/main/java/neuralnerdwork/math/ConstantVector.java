package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

import java.util.Arrays;

public record ConstantVector(double[] values) implements Vector {

    @Override
    public double get(int index) {
        return values[index];
    }

    @Override
    public int length() {
        return values.length;
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(values)
                     .allMatch(n -> n == 0.0);
    }

    @Override
    public Vector evaluate(Model.ParameterBindings bindings) {
        return this;
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new ConstantVector(new double[values.length]);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new DMatrixSparseCSC(values.length, bindings.size(), 0);
    }

    @Override
    public double[] toArray() {
        return values;
    }

    @Override
    public String toString() {
        return "ConstantVector{" +
                "values=" + Arrays.toString(values) +
                '}';
    }
}
