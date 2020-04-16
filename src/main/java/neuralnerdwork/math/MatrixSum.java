package neuralnerdwork.math;

import java.util.Map;

public record MatrixSum(MatrixExpression left, MatrixExpression right) implements MatrixExpression {
    public static MatrixExpression sum(MatrixExpression left, MatrixExpression right) {
        final MatrixSum fullSum = new MatrixSum(left, right);
        if (fullSum.isZero()) {
            return new SparseConstantMatrix(Map.of(), fullSum.rows(), fullSum.cols());
        } else if (left.isZero()) {
            return right;
        } else if (right.isZero()) {
            return left;
        } else {
            return fullSum;
        }
    }

    public MatrixSum {
        if (left.rows() != right.rows() || left.cols() != right.cols()) {
            throw new IllegalArgumentException(String.format("Cannot add matrices of dimensions (%dx%d) and (%dx%d)",
                                                             left.rows(),
                                                             left.cols(),
                                                             right.rows(),
                                                             right.cols()));
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() && right.isZero();
    }

    @Override
    public int rows() {
        return left.rows();
    }

    @Override
    public int cols() {
        return right.cols();
    }

    @Override
    public Matrix evaluate(Model.Binder bindings) {
        final Matrix leftMatrix = left.evaluate(bindings);
        final Matrix rightMatrix = right.evaluate(bindings);
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
    public MatrixExpression computePartialDerivative(int variable) {
        return MatrixSum.sum(left.computePartialDerivative(variable), right.computePartialDerivative(variable));
    }
}
