package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public class MatrixEqualityComparator {
    public boolean equal(DMatrix left, DMatrix right, double epsilon) {
        if (left.getNumRows() != right.getNumRows() || left.getNumCols() != right.getNumCols()) {
            return false;
        } else {
            for (int i = 0; i < left.getNumRows(); i++) {
                for (int j = 0; j < left.getNumCols(); j++) {
                    if (Math.abs(left.get(i, j) - right.get(i, j)) >= epsilon) {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}
