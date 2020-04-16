package neuralnerdwork.math;

import java.util.Arrays;

public record ConstantMatrix(double[][] values) implements MatrixExpression, Matrix {
    @Override
    public double get(int row, int col) {
        return values[row][col];
    }

    @Override
    public int rows() {
        return values.length;
    }

    @Override
    public int cols() {
        return (values.length > 0) ?
                values[0].length :
                0;
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(values)
                     .flatMapToDouble(Arrays::stream)
                     .allMatch(n -> n == 0.0);
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        return this;
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return new ConstantMatrix(new double[rows()][cols()]);
    }

    @Override
    public String toString() {
        return "ConstantMatrix{" +
                "values=" + Arrays.deepToString(values) +
                '}';
    }
}
