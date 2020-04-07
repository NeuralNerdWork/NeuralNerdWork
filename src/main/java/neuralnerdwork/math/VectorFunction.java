package neuralnerdwork.math;

public interface VectorFunction extends Differentiable {
    Vector apply(ScalarVariableBinding[] input);
    VectorFunction differentiate(ScalarVariable variable);
    MatrixFunction differentiate(ScalarVariable[] variable);
}
