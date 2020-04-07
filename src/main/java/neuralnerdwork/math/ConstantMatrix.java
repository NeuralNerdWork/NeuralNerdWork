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
    public Matrix apply(ScalarVariableBinding[] input) {
        return this;
    }

    @Override
    public MatrixFunction differentiate(ScalarVariable variable) {
        return new ConstantMatrix(new double[rows()][cols()]);
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Set.of();
    }

    @Override
    public String toString() {
        return "ConstantMatrix{" +
                "values=" + Arrays.deepToString(values) +
                '}';
    }
}
