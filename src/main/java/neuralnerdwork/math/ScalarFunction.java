package neuralnerdwork.math;

public interface ScalarFunction extends Differentiable {
    double apply(ScalarVariableBinding[] input);
    ScalarFunction differentiate(ScalarVariable variable);
    VectorFunction differentiate(ScalarVariable[] variable);
}
