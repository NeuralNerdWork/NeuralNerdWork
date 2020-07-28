package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;

import java.util.stream.IntStream;

public record ParameterMatrix(int variableStartIndex, int rows, int cols) implements MatrixExpression {
    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final int length = this.rows * this.cols;
        final DMatrixRMaj values = new DMatrixRMaj(this.rows, this.cols);
        for (int i = 0; i < length; i++) {
            final int row = i / this.cols;
            final int col = i % this.cols;

            values.set(row, col, bindings.get(this.variableStartIndex + i));
        }

        return values;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final int length = rows * cols;
        if (variable >= variableStartIndex && variable < variableStartIndex + length) {
            final int row = (variable - variableStartIndex) / cols;
            final int col = (variable - variableStartIndex) % cols;
            DMatrixSparseCSC retVal = new DMatrixSparseCSC(rows, cols, 1);
            retVal.set(row, col, 1.0);
            return retVal;
        }
        return new DMatrixSparseCSC(rows, cols, 0);
    }

    public IntStream variables() {
        return IntStream.iterate(variableStartIndex, i -> i < variableStartIndex + rows * cols, i -> i + 1);
    }

    public int variableIndexFor(int row, int col) {
        return variableStartIndex + row * cols + col;
    }

    public boolean containsVariable(int variable) {
        return variable >= variableStartIndex && variable < variableStartIndex + rows * cols;
    }

    public int rowIndexFor(int variable) {
        return (variable - variableStartIndex) / cols;
    }

    public int colIndexFor(int variable) {
        return (variable - variableStartIndex) % cols;
    }
}
