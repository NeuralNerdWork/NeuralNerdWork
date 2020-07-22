package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

public record MatrixSum(MatrixExpression... expressions) implements MatrixExpression {
    public static MatrixExpression sum(MatrixExpression... expressions) {
        MatrixExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(MatrixExpression[]::new);
        if (nonZeroExpressions.length == 0) {
            return new SparseConstantMatrix(Map.of(), expressions[0].rows(), expressions[0].cols());
        } else {
            return new MatrixSum(nonZeroExpressions);
        }
    }

    public MatrixSum {
        record Size(int rows, int cols) {}
        if (Arrays.stream(expressions)
                  .map(exp -> new Size(exp.rows(), exp.cols()))
                  .distinct().count() != 1) {
            String dimString = Arrays.stream(expressions)
                                     .map(exp -> new Size(exp.rows(), exp.cols()))
                                     .distinct()
                                     .map(Object::toString)
                                     .collect(Collectors.joining(",", "[", "]"));
            throw new IllegalArgumentException(String.format("Cannot add matrices of dimensions %s", dimString));
        }
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(MatrixExpression::isZero);
    }

    @Override
    public int rows() {
        return expressions[0].rows();
    }

    @Override
    public int cols() {
        return expressions[0].cols();
    }

    @Override
    public Matrix evaluate(Model.ParameterBindings bindings) {
        Matrix[] evaluated = Arrays.stream(expressions)
                                   .map(exp -> exp.evaluate(bindings))
                                   .toArray(Matrix[]::new);

        return new Matrix() {
            @Override
            public double get(int row, int col) {
                return Arrays.stream(evaluated)
                             .mapToDouble(m -> m.get(row, col))
                             .sum();
            }

            @Override
            public int rows() {
                return evaluated[0].rows();
            }

            @Override
            public int cols() {
                return evaluated[0].cols();
            }
        };
    }

    @Override
    public Matrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return MatrixSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(bindings, variable))
                      .toArray(MatrixExpression[]::new)
        ).evaluate(bindings);
    }
}
