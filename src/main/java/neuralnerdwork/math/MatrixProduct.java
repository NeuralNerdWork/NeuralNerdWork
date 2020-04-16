package neuralnerdwork.math;

import java.util.Map;

public record MatrixProduct(MatrixExpression left, MatrixExpression right) implements MatrixExpression {
    public static MatrixExpression product(MatrixExpression left, MatrixExpression right) {
        final MatrixProduct product = new MatrixProduct(left, right);
        if (product.isZero()) {
            return new SparseConstantMatrix(Map.of(), product.rows(), product.cols());
        } else {
            return product;
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
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
    public MatrixExpression computePartialDerivative(int variable) {
        // This uses product rule
        // (FG)' = F'G + FG'
        final MatrixExpression leftDerivative = left.computePartialDerivative(variable);
        final MatrixExpression rightDerivative = right.computePartialDerivative(variable);

        return MatrixSum.sum(
                MatrixProduct.product(leftDerivative, right),
                MatrixProduct.product(left, rightDerivative)
        );
    }
}
