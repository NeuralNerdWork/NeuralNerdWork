package neuralnerdwork.math;

import java.util.Arrays;

public record RepeatedScalarVector(double value, int length) implements Vector {
    @Override
    public double get(int index) {
        return value;
    }

    @Override
    public double[] toArray() {
        double[] values = new double[length];
        Arrays.fill(values, value);

        return values;
    }

    @Override
    public boolean isZero() {
        return value == 0.0;
    }
}
