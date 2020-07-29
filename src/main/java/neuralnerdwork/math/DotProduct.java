package neuralnerdwork.math;

import org.ejml.data.DMatrix;

public record DotProduct(VectorExpression left, VectorExpression right) implements ScalarExpression {
    public static ScalarExpression product(VectorExpression left, VectorExpression right) {
        final DotProduct product = new DotProduct(left, right);
        if (product.isZero()) {
            return new ConstantScalar(0.0);
        } else {
            return product;
        }
    }

    @Override
    public boolean isZero() {
        return left.isZero() || right.isZero();
    }

    @Override
    public double evaluate(Model.ParameterBindings bindings) {
        final DMatrix lVector = left.evaluate(bindings);
        final DMatrix rVector = right.evaluate(bindings);

        double accum = 0.0;
        // TODO check matching length and throw with message
        for (int i = 0; i < Math.max(lVector.getNumRows(), rVector.getNumRows()); i++) {
            accum += lVector.get(i, 0) * rVector.get(i, 0);
        }

        return accum;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);
        DMatrix leftValue = left.evaluate(bindings);
        DMatrix rightValue = right.evaluate(bindings);

        return MatrixSum.sum(
                new MatrixProduct(new TransposeExpression(new DMatrixExpression(leftValue)), new DMatrixExpression(rightDerivative)),
                new MatrixProduct(new TransposeExpression(new DMatrixExpression(rightValue)), new DMatrixExpression(leftDerivative))
        ).evaluate(bindings);
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final DMatrix leftDerivative = left.computePartialDerivative(bindings, variable);
        final DMatrix rightDerivative = right.computePartialDerivative(bindings, variable);

        return ScalarSum.sum(
                DotProduct.product(new DMatrixColumnVectorExpression(leftDerivative), right),
                DotProduct.product(left, new DMatrixColumnVectorExpression(rightDerivative))
        ).evaluate(bindings);
    }
}
