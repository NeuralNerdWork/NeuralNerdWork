package neuralnerdwork.math;

public record MatrixCompose(VectorFunction vector, MatrixFunction matrix) implements MatrixFunction {
    @Override
    public int rows() {
        return matrix.rows();
    }

    @Override
    public int cols() {
        return matrix.cols();
    }

    @Override
    public int inputLength() {
        return vector.inputLength();
    }

    @Override
    public Matrix apply(double[] inputs) {
        return matrix.apply(vector.apply(inputs).toArray());
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }
}
