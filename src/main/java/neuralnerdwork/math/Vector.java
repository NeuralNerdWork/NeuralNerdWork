package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

public interface Vector extends VectorExpression {
    double get(int index);
    int length();
    
    default double lOneNorm() {
        double sum = 0.0;
        for (int i = 0; i < length(); i++) {
            sum += Math.abs(get(i));
        }
        return sum;
    }

    default double lTwoNorm() {
        double sum = 0.0;
        for (int i = 0; i < length(); i++) {
            double component = get(i);
            sum += component * component;
        }
        return Math.sqrt(sum);
    }

    double[] toArray();

    @Override
    default Vector evaluate(Model.ParameterBindings bindings) {
        return this;
    }

    @Override
    default Vector computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return new RepeatedScalarVector(0, length());
    }

    @Override
    default DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return new DMatrixSparseCSC(length(), bindings.size(), 0);
    }
}
