package neuralnerdwork.math;

public interface VectorFunction {
    /**
     * @return Length of output vector
     */
    int length();

    Vector apply(double[] inputs);
    VectorFunction differentiate(int variableIndex);
    int inputLength();

    default MatrixFunction differentiate() {
        final VectorFunction[] columns = new VectorFunction[inputLength()];
        for (int i = 0; i < inputLength(); i++) {
            columns[i] = differentiate(i);
        }

        return new ColumnMatrixFunction(columns);
    }
}
