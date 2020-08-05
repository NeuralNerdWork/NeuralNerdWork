package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

import neuralnerdwork.math.Model.ParameterBindings;

public record SquareRoot(ScalarExpression expression) implements ScalarExpression {

    @Override
    public double evaluate(ParameterBindings bindings) {
        double inner = expression.evaluate(bindings);
        return Math.sqrt(inner);
    }

    @Override
    public double computePartialDerivative(ParameterBindings bindings, int variable) {
        double innerDerivative = expression.computePartialDerivative(bindings, variable);
        double innerValue = expression.evaluate(bindings);

        return (0.5 * Math.pow(innerValue, -0.5)) * innerDerivative;
    }

    @Override
    public boolean isZero() {
        return expression.isZero();
    }

    @Override
    public DMatrix computeDerivative(ParameterBindings bindings) {
        double[] values = new double[bindings.size()];
        int i = 0;
        for (int variable : bindings.variables()) {
            values[i++] = computePartialDerivative(bindings, variable); 
        }

        return new DMatrixRMaj(1, bindings.size(), true, values);
    }
    
}