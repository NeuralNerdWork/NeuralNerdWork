package neuralnerdwork.math;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.ejml.data.DMatrix;

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
        public double evaluate(Model.ParameterBindings bindings) {
            return function.apply(inputExpression.evaluate(bindings));
        }

        @Override
        public double computePartialDerivative(Model.ParameterBindings bindings, int variable) {
            return ScalarProduct
                    .product(function.differentiateByInput().invoke(inputExpression), new ConstantScalar(inputExpression
                                                                                                                 .computePartialDerivative(bindings, variable)))
                    .evaluate(bindings);
        }

        @Override
        public DMatrix computeDerivative(Model.ParameterBindings bindings) {
            DMatrix innerDerivative = inputExpression.computeDerivative(bindings);
            SingleVariableFunction outerDerivative = function.differentiateByInput();

            return new ScaledVector(
                    outerDerivative.invoke(inputExpression),
                    new DMatrixColumnVectorExpression(innerDerivative)
            ).evaluate(bindings);
        }

        @Override
        public boolean isZero() {
            return false;
        }
    }
}
