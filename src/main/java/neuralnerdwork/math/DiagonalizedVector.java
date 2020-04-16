package neuralnerdwork.math;

import java.util.HashMap;
import java.util.Map;

public record DiagonalizedVector(VectorExpression vector) implements MatrixExpression {
    @Override
    public int rows() {
        return vector.length();
    }

    @Override
    public int cols() {
        return vector.length();
    }

    @Override
    public boolean isZero() {
        return vector.isZero();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Vector vectorValue = vector.evaluate(bindings);
        final Map<SparseConstantMatrix.Index, Double> values = new HashMap<>();
        for (int i = 0; i < vector.length(); i++) {
            values.put(new SparseConstantMatrix.Index(i, i), vectorValue.get(i));
        }

        return new SparseConstantMatrix(values, rows(), cols());
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new DiagonalizedVector(vector.computePartialDerivative(variable));
    }
}
