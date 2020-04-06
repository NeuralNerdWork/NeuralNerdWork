package neuralnerdwork.math;

import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toSet;

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
    public Matrix apply(VectorVariableBinding input) {
        final Matrix leftMatrix = left.apply(input);
        final Matrix rightMatrix = right.apply(input);
        assert leftMatrix.cols() == rightMatrix.rows() :
                String.format("Cannot multiply matrices of dimentions (%dx%d) and (%dx%d)",
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
    public MatrixFunction differentiate(ScalarVariable variable) {
        final boolean leftContains = left.variables().contains(variable);
        final boolean rightContains = right.variables().contains(variable);

        if (leftContains && rightContains) {
            final MatrixFunction leftDerivative = left.differentiate(variable);
            final MatrixFunction rightDerivative = right.differentiate(variable);

            return new MatrixSumFunction(
                    new MatrixMultiplyFunction(leftDerivative, right),
                    new MatrixMultiplyFunction(left, rightDerivative)
            );
        } else if (leftContains) {
            final MatrixFunction leftDerivative = left.differentiate(variable);
            return new MatrixMultiplyFunction(
                    leftDerivative,
                    right
            );
        } else if (rightContains) {
            final MatrixFunction rightDerivative = right.differentiate(variable);
            return new MatrixMultiplyFunction(
                    left,
                    rightDerivative
            );
        } else {
            return new SparseConstantMatrix(Map.of(), rows(), cols());
        }
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.concat(left.variables()
                                 .stream(),
                             right.variables()
                                  .stream())
                     .collect(toSet());
    }
}
