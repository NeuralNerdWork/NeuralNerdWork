package neuralnerdwork.math;

public interface Vector {
    double get(int index);
    int length();
    default double[] toArray() {
        final double[] retVal = new double[length()];
        for (int i = 0; i < length(); i++) {
            retVal[i] = get(i);
        }

        return retVal;
    }
}
