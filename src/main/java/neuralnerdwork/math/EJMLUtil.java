package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

public class EJMLUtil {
    static DMatrix mult(DMatrix leftMatrix, DMatrix rightMatrix) {
        if (leftMatrix instanceof DMatrixRMaj l && rightMatrix instanceof DMatrixRMaj r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DDRM.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixSparseCSC l && rightMatrix instanceof DMatrixSparseCSC r) {
            DMatrixSparseCSC retVal = new DMatrixSparseCSC(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixSparseCSC l && rightMatrix instanceof DMatrixRMaj r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixRMaj l && rightMatrix instanceof DMatrixSparseCSC r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.multTransAB(r, l, retVal);
            CommonOps_DDRM.transpose(retVal);

            return retVal;
        } else {
            throw new UnsupportedOperationException("Cannot multiply matrix types " + leftMatrix.getClass() + " and " + rightMatrix.getClass());
        }
    }
}
