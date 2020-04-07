package neuralnerdwork.math;

public interface MatrixFunction extends Differentiable {
    int rows();
    int cols();
    Matrix apply(ScalarVariableBinding[] input);
    MatrixFunction differentiate(ScalarVariable argument);
}
