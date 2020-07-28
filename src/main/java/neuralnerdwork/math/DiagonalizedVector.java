package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DMatrixSparseTriplet;
import org.ejml.ops.ConvertDMatrixStruct;

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
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final Vector vectorValue = vector.evaluate(bindings);
        DMatrixSparseTriplet sparseBuilder = new DMatrixSparseTriplet(rows(), cols(), vector.length());
        for (int i = 0; i < vector.length(); i++) {
            sparseBuilder.addItem(i, i, vectorValue.get(i));
        }

        return ConvertDMatrixStruct.convert(sparseBuilder, (DMatrixSparseCSC) null);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new DiagonalizedVector(vector.computePartialDerivative(bindings, variable)).evaluate(bindings);
    }
}
