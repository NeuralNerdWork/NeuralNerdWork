package neuralnerdwork.math;

public record MatrixSumFunction(MatrixFunction left, MatrixFunction right) implements MatrixFunction {
    public MatrixSumFunction {
        if (left.rows() != right.rows() || left.cols() != right.cols()) {
            throw new IllegalArgumentException(String.format("Cannot add matrices of dimensions (%dx%d) and (%dx%d)",
                                                             left.rows(),
                                                             left.cols(),
                                                             right.rows(),
                                                             right.cols()));
        }
    }

    @Override
    public int inputLength() {
        return Math.max(left.inputLength(), right.inputLength());
    }

    @Override
    public int rows() {
        return left.rows();
    }

    @Override
    public int cols() {
        return right.rows();
    }

    @Override
    public Matrix apply(double[] input) {
        final Matrix leftMatrix = left.apply(input);
        final Matrix rightMatrix = right.apply(input);
        assert leftMatrix.rows() == rightMatrix.rows() && leftMatrix.cols() == rightMatrix.cols() :
                String.format("Cannot add matrices of dimensions (%dx%d) and (%dx%d)",
                              leftMatrix.rows(),
                              leftMatrix.cols(),
                              rightMatrix.rows(),
                              rightMatrix.cols());

        return new Matrix() {
            @Override
            public double get(int row, int col) {
                return leftMatrix.get(row, col) + rightMatrix.get(row, col);
            }

            @Override
            public int rows() {
                return leftMatrix.rows();
            }

            @Override
            public int cols() {
                return leftMatrix.cols();
            }
        };
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        return new MatrixSumFunction(left.differentiate(variableIndex), right.differentiate(variableIndex));
    }
}
