package neuralnerdwork.math;

public class MatrixEqualityComparator {
    public boolean equal(Matrix left, Matrix right, double epsilon) {
        if (left.rows() != right.rows() || left.cols() != right.cols()) {
            return false;
        } else {
            for (int i = 0; i < left.rows(); i++) {
                for (int j = 0; j < left.cols(); j++) {
                    if (Math.abs(left.get(i, j) - right.get(i, j)) >= epsilon) {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}
