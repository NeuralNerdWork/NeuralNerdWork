package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.sparse.csc.CommonOps_DSCC;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Collectors;

public record MatrixSum(MatrixExpression... expressions) implements MatrixExpression {
    public static MatrixExpression sum(MatrixExpression... expressions) {
        MatrixExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(MatrixExpression[]::new);
        if (nonZeroExpressions.length == 0) {
            DMatrixSparseCSC zeroMatrix = new DMatrixSparseCSC(expressions[0].rows(), expressions[0].cols(), 0);
            return new DMatrixExpression(zeroMatrix);
        } else {
            return new MatrixSum(nonZeroExpressions);
        }
    }

    public MatrixSum {
        record Size(int rows, int cols) {}
        if (Arrays.stream(expressions)
                  .map(exp -> new Size(exp.rows(), exp.cols()))
                  .distinct().count() != 1) {
            String dimString = Arrays.stream(expressions)
                                     .map(exp -> new Size(exp.rows(), exp.cols()))
                                     .distinct()
                                     .map(Object::toString)
                                     .collect(Collectors.joining(",", "[", "]"));
            throw new IllegalArgumentException(String.format("Cannot add matrices of dimensions %s", dimString));
        }
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(MatrixExpression::isZero);
    }

    @Override
    public int rows() {
        return expressions[0].rows();
    }

    @Override
    public int cols() {
        return expressions[0].cols();
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        DMatrixRMaj retVal = new DMatrixRMaj(rows(), cols());
        for (int i = 0; i < expressions.length; i++) {
            DMatrix evaluated = expressions[i].evaluate(bindings);
            if (evaluated instanceof DMatrixRMaj m) {
                CommonOps_DDRM.add(retVal, m, retVal);
            } else if (evaluated instanceof DMatrixSparseCSC m) {
                Iterator<DMatrixSparse.CoordinateRealValue> coords = m.createCoordinateIterator();
                while (coords.hasNext()) {
                    DMatrixSparse.CoordinateRealValue coord = coords.next();
                    retVal.add(coord.row, coord.col, coord.value);
                }
            } else {
                throw new UnsupportedOperationException("Cannot add matrix of type " + evaluated.getClass());
            }
        }

        return retVal;
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return MatrixSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(bindings, variable))
                      .toArray(MatrixExpression[]::new)
        ).evaluate(bindings);
    }
}
