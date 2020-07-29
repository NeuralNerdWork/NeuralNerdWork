package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record VectorSum(VectorExpression... expressions) implements VectorExpression {
    public static VectorExpression sum(VectorExpression... expressions) {
        VectorExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(VectorExpression[]::new);
        if (nonZeroExpressions.length == 0) {
            return new DMatrixColumnVectorExpression(new DMatrixSparseCSC(expressions[0].length(), 1, 0));
        } else {
            return new VectorSum(nonZeroExpressions);
        }
    }

    public VectorSum {
        if (Arrays.stream(expressions).mapToInt(VectorExpression::length).distinct().count() != 1) {
            throw new IllegalArgumentException("Cannot add vectors of different lengths: " + Arrays.stream(expressions)
                                                                                                   .distinct()
                                                                                                   .map(VectorExpression::length)
                                                                                                   .map(Object::toString)
                                                                                                   .collect(Collectors.joining(",")));
        }
    }

    @Override
    public int length() {
        return expressions[0].length();
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(VectorExpression::isZero);
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        DMatrix[] evaluated = Arrays.stream(expressions)
                                    .map(exp -> exp.evaluate(bindings))
                                    .toArray(DMatrix[]::new);

        final int length = length();
        final double[] values = IntStream.iterate(0, i -> i < length, i -> i + 1)
                                         .mapToDouble(i -> Arrays.stream(evaluated)
                                                                 .mapToDouble(components -> {
                                                                     if (components.getNumRows() > 1) {
                                                                         return components.get(i, 0);
                                                                     } else {
                                                                         return components.get(0, i);
                                                                     }
                                                                 })
                                                                 .sum())
                                         .toArray();

        return new DMatrixRMaj(values);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return MatrixSum.sum(Arrays.stream(expressions)
                                   .map(exp -> exp.computeDerivative(bindings))
                                   .map(DMatrixExpression::new)
                                   .toArray(MatrixExpression[]::new))
                        .evaluate(bindings);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(bindings, variable))
                      .map(DMatrixColumnVectorExpression::new)
                      .toArray(VectorExpression[]::new)
        ).evaluate(bindings);
    }
}
