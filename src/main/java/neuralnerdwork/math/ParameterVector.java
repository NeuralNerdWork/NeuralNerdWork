package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;

import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

public record ParameterVector(int variableStartIndex, int length) implements VectorExpression {

    @Override
    public boolean columnVector() {
        return true;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = bindings.get(variableStartIndex + i);
        }

        return new DMatrixRMaj(length, 1, true, values);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        if (variable >= variableStartIndex && variable < variableStartIndex + length) {
            final double[] values = new double[length];
            values[variable - variableStartIndex] = 1.0;

            return new DMatrixRMaj(length, 1, true, values);
        } else {
            return new DMatrixSparseCSC(length, 1, 0);
        }
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new ColumnMatrix(
                StreamSupport.stream(bindings.variables().spliterator(), false)
                             .map(variable -> computePartialDerivative(bindings, variable))
                             .map(DMatrixColumnVectorExpression::new)
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
