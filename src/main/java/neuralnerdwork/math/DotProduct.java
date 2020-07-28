package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

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
        final Vector lVector = left.evaluate(bindings);
        final Vector rVector = right.evaluate(bindings);

        double accum = 0.0;
        // TODO check matching length and throw with message
        for (int i = 0; i < Math.max(lVector.length(), rVector.length()); i++) {
            accum += lVector.get(i) * rVector.get(i);
        }

        return accum;
    }

    @Override
    public Vector computeDerivative(Model.ParameterBindings bindings) {
        final DMatrix leftDerivative = left.computeDerivative(bindings);
        final DMatrix rightDerivative = right.computeDerivative(bindings);
        Vector leftValue = left.evaluate(bindings);
        Vector rightValue = right.evaluate(bindings);

        DMatrixRMaj leftAsMatrix = new DMatrixRMaj(1, leftValue.length(), true, leftValue.toArray());
        DMatrixRMaj rightAsMatrix = new DMatrixRMaj(1, rightValue.length(), true, rightValue.toArray());

        // FIXME Use vector sum after getting rid of vectors
        DMatrix evaluate = MatrixSum.sum(
                new MatrixProduct(new DMatrixExpression(leftAsMatrix), new DMatrixExpression(leftDerivative)),
                new MatrixProduct(new DMatrixExpression(rightAsMatrix), new DMatrixExpression(rightDerivative))
        ).evaluate(bindings);

        return new DMatrixRowVector(evaluate);
    }

    @Override
    public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final Vector leftDerivative = left.computePartialDerivative(bindings, variable);
        final Vector rightDerivative = right.computePartialDerivative(bindings, variable);

        return ScalarSum.sum(
                DotProduct.product(leftDerivative, right),
                DotProduct.product(left, rightDerivative)
        ).evaluate(bindings);
    }
}
