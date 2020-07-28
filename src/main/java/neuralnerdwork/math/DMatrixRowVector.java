package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public record DMatrixRowVector(DMatrix matrix) implements Vector {
    @Override
    public double get(int index) {
        return matrix.get(0, index);
    }

    @Override
    public int length() {
        return matrix.getNumCols();
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public double[] toArray() {
        double[] values = new double[length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = get(i);
        }

        return values;
    }
}
