package neuralnerdwork.math;

public interface MatrixFunction extends Differentiable {
    int rows();
    int cols();
    Matrix apply(VectorVariableBinding input);
    MatrixFunction differentiate(ScalarVariable argument);
}
