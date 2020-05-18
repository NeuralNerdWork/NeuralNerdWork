package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarSum(ScalarExpression... expressions) implements ScalarExpression {
    public static ScalarExpression sum(ScalarExpression... expressions) {
        ScalarExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(ScalarExpression[]::new);

        if (nonZeroExpressions.length == 0) {
            return new ConstantScalar(0.0);
        } else {
            return new ScalarSum(nonZeroExpressions);
        }
    }

    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        return Arrays.stream(expressions)
                     .mapToDouble(exp -> exp.evaluate(bindings))
                     .sum();
    }

    @Override
    public ScalarExpression computePartialDerivative(int variable) {
        return ScalarSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(variable))
                      .toArray(ScalarExpression[]::new)
        );
    }

    @Override
    public VectorExpression computeDerivative(int[] variables) {
        return VectorSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computeDerivative(variables))
                      .toArray(VectorExpression[]::new)
        );
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(ScalarExpression::isZero);
    }
}
