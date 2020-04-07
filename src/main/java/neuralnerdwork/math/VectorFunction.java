package neuralnerdwork.math;

public interface VectorFunction {
    /**
     * @return Length of output vector
     */
    int length();

    Vector apply(double[] inputs);
    VectorFunction differentiate(int variableIndex);

    default MatrixFunction differentiate() {
        final VectorFunction[] columns = new VectorFunction[length()];
        for (int i = 0; i < length(); i++) {
            columns[i] = differentiate(i);
        }

        return new ColumnMatrixFunction(columns);
    }
}
