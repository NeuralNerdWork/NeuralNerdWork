package neuralnerdwork.math;

import java.util.Arrays;
import java.util.function.ObjIntConsumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record VectorSum(VectorExpression... expressions) implements VectorExpression {
    public static VectorExpression sum(VectorExpression... expressions) {
        VectorExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(VectorExpression[]::new);
        if (nonZeroExpressions.length == 0) {
            return new ConstantVector(new double[expressions[0].length()]);
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
    public Vector evaluate(Model.ParameterBindings bindings) {
        Vector[] evaluated = Arrays.stream(expressions)
                                   .map(exp -> exp.evaluate(bindings))
                                   .toArray(Vector[]::new);

        final int length = length();
        final double[] values = IntStream.iterate(0, i -> i < length, i -> i + 1)
                                         .mapToDouble(i -> Arrays.stream(evaluated)
                                                                 .mapToDouble(components -> components.get(i))
                                                                 .sum())
                                         .toArray();

        return new ConstantVector(values);
    }

    @Override
    public Matrix computeDerivative(Model.ParameterBindings bindings, int[] variables) {
        return MatrixSum.sum(Arrays.stream(expressions)
                                   .map(exp -> exp.computeDerivative(bindings, variables))
                                   .toArray(MatrixExpression[]::new)).evaluate(bindings);
    }

    @Override
    public Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(bindings, variable))
                      .toArray(VectorExpression[]::new)
        ).evaluate(bindings);
    }
}
