package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Set;

public record ConstantMatrix(double[][] values) implements MatrixFunction, Matrix {
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
    public int inputLength() {
        return 0;
    }

    @Override
    public Matrix apply(double[] inputs) {
        return this;
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        return new ConstantMatrix(new double[rows()][cols()]);
    }

    @Override
    public String toString() {
        return "ConstantMatrix{" +
                "values=" + Arrays.deepToString(values) +
                '}';
    }
}
