package neuralnerdwork.math;

import org.ejml.data.*;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

public class EJMLUtil {

    private static final ThreadLocal<IGrowArray> intWorkArrays = new ThreadLocal<>();
    private static final ThreadLocal<DGrowArray> doubleWorkArrays = new ThreadLocal<>();

    public static DMatrix mult(DMatrix leftMatrix, DMatrix rightMatrix) {
        if (leftMatrix instanceof DMatrixRMaj l && rightMatrix instanceof DMatrixRMaj r) {
            DMatrixRMaj retVal = new DMatrixRMaj(l.getNumRows(), r.getNumCols());
            CommonOps_DDRM.mult(l, r, retVal);

            return retVal;
        } else if (leftMatrix instanceof DMatrixSparseCSC l && rightMatrix instanceof DMatrixSparseCSC r) {
            IGrowArray intWorkArray = intWorkArrays.get();
            if (intWorkArray == null) {
                intWorkArray = new IGrowArray(15_000);
                intWorkArrays.set(intWorkArray);
            }
            DGrowArray doubleWorkArray = doubleWorkArrays.get();
            if (doubleWorkArray == null) {
                doubleWorkArray = new DGrowArray(15_000);
                doubleWorkArrays.set(doubleWorkArray);
            }
            DMatrixSparseCSC retVal = new DMatrixSparseCSC(l.getNumRows(), r.getNumCols());
            CommonOps_DSCC.mult(l, r, retVal, intWorkArray, doubleWorkArray);

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
