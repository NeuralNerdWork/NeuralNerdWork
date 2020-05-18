package neuralnerdwork.math;

import java.util.Map;

public record ConstantMatrixExpression(Matrix matrix) implements MatrixExpression {

    @Override
    public int rows() {
        return matrix.rows();
    }

    @Override
    public int cols() {
        return matrix.cols();
    }

    @Override
    public boolean isZero() {
        for (int row = 0; row < rows(); row++) {
            for (int col = 0; col < cols(); col++) {
                if (matrix.get(row, col) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
        return matrix;
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new SparseConstantMatrix(Map.of(), rows(), cols());
    }
}
