package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

import java.util.Arrays;

public record ColumnMatrix(VectorExpression[] columns) implements MatrixExpression {
    public ColumnMatrix {
        if (columns == null || columns.length == 0) {
            throw new IllegalArgumentException("Cannot have column matrix with empty columns");
        }
        final long count = Arrays.stream(columns)
                                 .map(VectorExpression::length)
                                 .distinct()
                                 .count();
        if (count != 1) {
            throw new IllegalArgumentException("All columns must be same length");
        }
    }

    public ColumnMatrix(VectorExpression column) {
        this(new VectorExpression[] {column});
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(columns)
                     .allMatch(VectorExpression::isZero);
    }

    @Override
    public int rows() {
        return columns[0].length();
    }

    @Override
    public int cols() {
        return columns.length;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrixRMaj values = new DMatrixRMaj(rows(), cols());
        for (int col = 0; col < cols(); col++) {
            final Vector vector = columns[col].evaluate(bindings);
            for (int row = 0; row < rows(); row++) {
                values.set(row, col, vector.get(row));
            }
        }

        return values;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public String toString() {
        return "ColumnMatrix[" +
                "columns=" + Arrays.toString(columns) +
                ']';
    }
}
