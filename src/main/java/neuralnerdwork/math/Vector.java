package neuralnerdwork.math;

public interface Vector extends VectorExpression {
    double get(int index);
    int length();
    
    default double lOneNorm() {
        double sum = 0.0;
        for (int i = 0; i < length(); i++) {
            sum += Math.abs(get(i));
        }
        return sum;
    }

    default double lTwoNorm() {
        double sum = 0.0;
        for (int i = 0; i < length(); i++) {
            double component = get(i);
            sum += component * component;
        }
        return Math.sqrt(sum);
    }

    default double[] toArray() {
        final double[] retVal = new double[length()];
        for (int i = 0; i < length(); i++) {
            retVal[i] = get(i);
        }

        return retVal;
    }
}
