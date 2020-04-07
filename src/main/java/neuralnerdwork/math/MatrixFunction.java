package neuralnerdwork.math;

public interface MatrixFunction {
    int rows();
    int cols();
    Matrix apply(double[] inputs);
    MatrixFunction differentiate(int variableIndex);
    int inputLength();
}
