package neuralnerdwork.math;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record VectorSumFunction(VectorFunction left, VectorFunction right) implements VectorFunction {
    @Override
    public Vector apply(VectorVariableBinding input) {
        final Vector left = this.left.apply(input);
        final Vector right = this.right.apply(input);
        final int length = Math.max(left.length(), right.length());

        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = left.get(i) + right.get(i);
        }

        return new ConstantVector(values);
    }

    @Override
    public VectorFunction differentiate(ScalarVariable variable) {
        return new VectorSumFunction(left.differentiate(variable), right.differentiate(variable));
    }

    @Override
    public MatrixFunction differentiate(VectorVariable variable) {
        return new MatrixSumFunction(left.differentiate(variable), right.differentiate(variable));
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
