package neuralnerdwork.math;

public interface ScalarFunction extends Differentiable {
    double apply(VectorVariableBinding input);
    ScalarFunction differentiate(ScalarVariable variable);
    VectorFunction differentiate(VectorVariable variable);
}
