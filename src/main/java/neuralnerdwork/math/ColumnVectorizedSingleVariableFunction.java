package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

public record ColumnVectorizedSingleVariableFunction(SingleVariableFunction function, VectorExpression vectorExpression) implements VectorExpression {
    public ColumnVectorizedSingleVariableFunction {
        if (!vectorExpression.columnVector()) {
            throw new IllegalArgumentException("Cannot vectorize a row vector expression");
        }
    }

    @Override
    public int length() {
        return vectorExpression.length();
    }

    @Override
    public boolean columnVector() {
        return true;
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
                new ColumnVectorizedSingleVariableFunction(
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
                new ColumnVectorizedSingleVariableFunction(function.differentiateByInput(),
                                                           vectorExpression),
                new DMatrixColumnVectorExpression(innerDerivative)).evaluate(bindings);
    }
}
