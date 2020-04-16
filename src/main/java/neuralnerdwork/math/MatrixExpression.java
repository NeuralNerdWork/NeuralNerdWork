package neuralnerdwork.math;

public interface MatrixExpression {
    int rows();
    int cols();
    boolean isZero();

    Matrix evaluate(Model.Binder bindings);
    MatrixExpression computePartialDerivative(int variable);
}
