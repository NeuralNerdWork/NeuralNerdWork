package neuralnerdwork.math;

import org.ejml.data.DMatrix;

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
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return ScalarSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> new ConstantScalar(exp.computePartialDerivative(bindings, variable)))
                      .toArray(ScalarExpression[]::new)
        ).evaluate(bindings);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return VectorSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computeDerivative(bindings))
                      .map(DMatrixRowVectorExpression::new)
                      .toArray(VectorExpression[]::new)
        ).evaluate(bindings);
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(ScalarExpression::isZero);
    }
}
