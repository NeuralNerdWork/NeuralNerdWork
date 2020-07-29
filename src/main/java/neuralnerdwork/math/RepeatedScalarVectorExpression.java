package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.ConvertDMatrixStruct;

import java.util.Arrays;

public record RepeatedScalarVectorExpression(ScalarExpression scalar, int length) implements VectorExpression {

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        double[] values = new double[length];
        Arrays.fill(values, scalar.evaluate(bindings));

        return new DMatrixRMaj(1, length, true, values);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        double[] values = new double[length];
        Arrays.fill(values, scalar.computePartialDerivative(bindings, variable));

        return new DMatrixRMaj(1, length, true, values);
    }

    @Override
    public boolean isZero() {
        return scalar.isZero();
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        DMatrix gradient = scalar.computeDerivative(bindings);
        if (gradient instanceof DMatrixSparseCSC m) {
            gradient = ConvertDMatrixStruct.convert(m, (DMatrixRMaj) null);
        }

        if (gradient instanceof DMatrixRMaj m) {
            DMatrixRMaj[] rows = new DMatrixRMaj[length];
            Arrays.fill(rows, m);
            return CommonOps_DDRM.concatRowsMulti(rows);
        } else {
            throw new UnsupportedOperationException("Cannot create repeated vector for gradient type " + gradient.getClass());
        }
    }
}
