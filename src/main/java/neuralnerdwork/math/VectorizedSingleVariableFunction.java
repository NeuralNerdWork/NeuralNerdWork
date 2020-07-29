package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record VectorizedSingleVariableFunction(SingleVariableFunction function, VectorExpression vectorExpression) implements VectorExpression {

    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        final DMatrix vector = vectorExpression.evaluate(bindings);
        final double[] values = new double[vector.getNumRows()];
        for (int i = 0; i < values.length; i++) {
            values[i] = function.apply(vector.get(i, 0));
        }

        return new DMatrixRMaj(values.length, 1, true, values);
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        final MatrixExpression outerDerivative = new DiagonalizedVector(
                new VectorizedSingleVariableFunction(
                        function.differentiateByInput(),
                        vectorExpression
                )
        );

        return MatrixProduct.product(outerDerivative, new DMatrixExpression(vectorExpression.computeDerivative(bindings)))
                            .evaluate(bindings);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        final DMatrix innerDerivative = this.vectorExpression.computePartialDerivative(bindings, variable);

        return VectorComponentProduct.product(
                new VectorizedSingleVariableFunction(function.differentiateByInput(),
                                                     vectorExpression),
                new DMatrixColumnVectorExpression(innerDerivative)).evaluate(bindings);
    }
}
