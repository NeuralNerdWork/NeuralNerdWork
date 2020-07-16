package neuralnerdwork.math;

import java.util.Map;

public interface Matrix extends MatrixExpression {
    double get(int row, int col);
    int rows();
    int cols();

    default double[][] toArray() {
        final double[][] values = new double[rows()][cols()];
        for (int row = 0; row < rows(); row++) {
            for (int col = 0; col < cols(); col++) {
                values[row][col] = get(row, col);
            }
        }

        return values;
    }

    @Override
    default boolean isZero() {
        for (int row = 0; row < rows(); row++) {
            for (int col = 0; col < cols(); col++) {
                if (get(row, col) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    default Matrix evaluate(Model.ParameterBindings bindings) {
        return this;
    }

    @Override
    default MatrixExpression computePartialDerivative(int variable) {
        return new SparseConstantMatrix(Map.of(), rows(), cols());
    }
}
