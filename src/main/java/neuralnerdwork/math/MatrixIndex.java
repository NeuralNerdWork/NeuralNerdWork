package neuralnerdwork.math;

public record MatrixIndex(int row, int col) implements Comparable<MatrixIndex> {

    @Override
    public int compareTo(MatrixIndex o) {
        if (row > o.row) {
            return 1;
        } else if (row < o.row) {
            return -1;
        } else if (col > o.col) {
            return 1;
        } else if (col < o.col) {
            return -1;
        } else {
            return 0;
        }
    }
}
