package neuralnerdwork.math;

public interface ScalarFunction {
    double apply(double[] inputs);
    ScalarFunction differentiate(int variableIndex);
    default VectorFunction differentiate() {
        throw new UnsupportedOperationException("Not yet implemented!");
    }
}
