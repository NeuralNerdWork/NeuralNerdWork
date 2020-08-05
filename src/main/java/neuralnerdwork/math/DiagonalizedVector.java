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
        final DMatrix vectorValue = vector.evaluate(bindings);
        DMatrixSparseTriplet sparseBuilder = new DMatrixSparseTriplet(rows(), cols(), vector.length());
        for (int i = 0; i < vector.length(); i++) {
            sparseBuilder.addItem(i, i, vectorValue.get(i, 0));
        }

        return ConvertDMatrixStruct.convert(sparseBuilder, (DMatrixSparseCSC) null);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        DMatrix vectorDerivative = vector.computePartialDerivative(bindings, variable);
        return new DiagonalizedVector(new DMatrixColumnVectorExpression(vectorDerivative)).evaluate(bindings);
    }
}
