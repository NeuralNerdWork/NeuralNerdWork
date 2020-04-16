package neuralnerdwork.math;

public interface VectorExpression {
    /**
     * @return Length of output vectorExpression
     */
    int length();

    Vector evaluate(Model.Binder bindings);
    VectorExpression computePartialDerivative(int variable);
    boolean isZero();

    default MatrixExpression computeDerivative(int[] variables) {
        final VectorExpression[] columns = new VectorExpression[variables.length];
        for (int i = 0; i < variables.length; i++) {
            columns[i] = computePartialDerivative(variables[i]);
        }

        return new ColumnMatrix(columns);
    }
}
