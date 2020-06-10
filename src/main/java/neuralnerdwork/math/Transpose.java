package neuralnerdwork.math;

public record Transpose(Matrix matrix) implements Matrix {
    @Override
    public double get(int row, int col) {
        return matrix.get(col, row);
    }

    @Override
    public int rows() {
        return matrix.cols();
    }

    @Override
    public int cols() {
        return matrix.rows();
    }
}
