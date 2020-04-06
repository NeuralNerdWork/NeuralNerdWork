package neuralnerdwork.math;

public interface VectorFunction extends Differentiable {
    Vector apply(VectorVariableBinding input);
    VectorFunction differentiate(ScalarVariable variable);
    MatrixFunction differentiate(VectorVariable variable);
}
