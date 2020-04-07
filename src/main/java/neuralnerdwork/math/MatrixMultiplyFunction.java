package neuralnerdwork.math;

public record MatrixMultiplyFunction(MatrixFunction left, MatrixFunction right) implements MatrixFunction {
    @Override
    public int rows() {
        return left.rows();
    }

    @Override
    public int cols() {
        return right.cols();
    }

    @Override
    public Matrix apply(double[] input) {
        final Matrix leftMatrix = left.apply(input);
        final Matrix rightMatrix = right.apply(input);
        assert leftMatrix.cols() == rightMatrix.rows() :
                String.format("Cannot multiply matrices of dimensions (%dx%d) and (%dx%d)",
                              leftMatrix.rows(),
                              leftMatrix.cols(),
                              rightMatrix.rows(),
                              rightMatrix.cols());

        final double[][] values = new double[leftMatrix.rows()][rightMatrix.cols()];
        for (int i = 0; i < leftMatrix.rows(); i++) {
            for (int j = 0; j < rightMatrix.cols(); j++) {
                for (int k = 0; k < leftMatrix.cols(); k++) {
                    values[i][j] += leftMatrix.get(i, k) * rightMatrix.get(k, j);
                }
            }
        }

        return new ConstantMatrix(values);
    }

    @Override
    public MatrixFunction differentiate(int variableIndex) {
        // This uses product rule
        // (FG)' = F'G + FG'
        // TODO maybe optimize for constant case?
        final MatrixFunction leftDerivative = left.differentiate(variableIndex);
        final MatrixFunction rightDerivative = right.differentiate(variableIndex);

        return new MatrixSumFunction(
                new MatrixMultiplyFunction(leftDerivative, right),
                new MatrixMultiplyFunction(left, rightDerivative)
        );
    }
}
