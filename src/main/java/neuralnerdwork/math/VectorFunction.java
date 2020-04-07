package neuralnerdwork.math;

public interface VectorFunction {
    Vector apply(double[] inputs);
    VectorFunction differentiate(int variableIndex);
    default MatrixFunction differentiate() {
        throw new UnsupportedOperationException("Not yet implemented!");
    }
}
