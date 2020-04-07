package neuralnerdwork.math;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record MatrixVectorMultFunction(MatrixFunction left, VectorFunction right) implements VectorFunction {
    @Override
    public Vector apply(ScalarVariableBinding[] input) {
        final Matrix leftValue = left.apply(input);
        final Vector rightValue = right.apply(input);
        assert leftValue.rows() == rightValue.length();

        final double[] values = new double[leftValue.rows()];
        for (int row = 0; row < leftValue.rows(); row++) {
            for (int col = 0; col < leftValue.cols(); col++) {
                values[row] += leftValue.get(row, col) * rightValue.get(col);
            }
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorFunction differentiate(ScalarVariable variable) {
        final boolean leftContains = left.variables().contains(variable);
        final boolean rightContains = right.variables().contains(variable);
        if (leftContains && rightContains) {
            throw new UnsupportedOperationException("Not yet implemented!");
        } else if (leftContains) {
            final MatrixFunction leftDerivative = left.differentiate(variable);
            return new MatrixVectorMultFunction(leftDerivative, right);
        } else if (rightContains) {
            final VectorFunction rightDerivative = right.differentiate(variable);
            return new MatrixVectorMultFunction(left, rightDerivative);
        } else {
            return new ConstantVector(new double[left.rows()]);
        }
    }

    @Override
    public MatrixFunction differentiate(ScalarVariable[] variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.concat(left.variables()
                                 .stream(),
                             right.variables()
                                  .stream())
                     .collect(Collectors.toSet());
    }
}
