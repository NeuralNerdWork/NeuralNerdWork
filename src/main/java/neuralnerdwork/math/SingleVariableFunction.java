package neuralnerdwork.math;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.Arrays;

public interface SingleVariableFunction {

    String getFunctionName();
    double apply(double input);
    SingleVariableFunction differentiateByInput();

    default ScalarExpression invoke(ScalarExpression inputExpression) {
        return new Invocation(inputExpression, this);
    }

    class Invocation implements ScalarExpression {
        private final ScalarExpression inputExpression;
        @JsonIgnore
        private final SingleVariableFunction function;

        public Invocation(ScalarExpression inputExpression, SingleVariableFunction function) {
            this.inputExpression = inputExpression;
            this.function = function;
        }

        @Override
        public double evaluate(Model.Binder bindings) {
            return function.apply(inputExpression.evaluate(bindings));
        }

        @Override
        public ScalarExpression computePartialDerivative(int variable) {
            return ScalarProduct.product(function.differentiateByInput().invoke(inputExpression), inputExpression.computePartialDerivative(variable));
        }

        @Override
        public VectorExpression computeDerivative(int[] variables) {
            VectorExpression innerDerivative = inputExpression.computeDerivative(variables);
            SingleVariableFunction outerDerivative = function.differentiateByInput();

            return new ScaledVector(
                    outerDerivative.invoke(inputExpression),
                    innerDerivative
            );
        }

        @Override
        public boolean isZero() {
            return false;
        }
    }
}
