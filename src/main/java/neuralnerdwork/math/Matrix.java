package neuralnerdwork.math;

public interface Matrix {
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
}
