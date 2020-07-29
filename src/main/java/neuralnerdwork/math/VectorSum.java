package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Collectors;

public record VectorSum(VectorExpression... expressions) implements VectorExpression {
    public static VectorExpression sum(VectorExpression... expressions) {
        VectorExpression[] nonZeroExpressions = Arrays.stream(expressions)
                                                      .filter(exp -> !exp.isZero())
                                                      .toArray(VectorExpression[]::new);
        if (nonZeroExpressions.length == 0) {
            return new DMatrixColumnVectorExpression(new DMatrixSparseCSC(expressions[0].length(), 1, 0));
        } else {
            return new VectorSum(nonZeroExpressions);
        }
    }

    public VectorSum {
        if (Arrays.stream(expressions).mapToInt(VectorExpression::length).distinct().count() != 1) {
            throw new IllegalArgumentException("Cannot add vectors of different lengths: " + Arrays.stream(expressions)
                                                                                                   .distinct()
                                                                                                   .map(VectorExpression::length)
                                                                                                   .map(Object::toString)
                                                                                                   .collect(Collectors.joining(",")));
        }
        if (Arrays.stream(expressions).map(VectorExpression::columnVector).distinct().count() != 1) {
            throw new IllegalArgumentException("Cannot add row and column vectors together: " + Arrays
                    .stream(expressions)
                    .distinct()
                    .map(VectorExpression::columnVector)
                    .map(Object::toString)
                    .collect(Collectors.joining(",")));
        }
    }

    @Override
    public int length() {
        return expressions[0].length();
    }

    @Override
    public boolean columnVector() {
        return expressions[0].columnVector();
    }

    @Override
    public boolean isZero() {
        return Arrays.stream(expressions)
                     .allMatch(VectorExpression::isZero);
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        boolean columnVector = columnVector();
        DMatrixRMaj accum = new DMatrixRMaj(columnVector ? length() : 1, columnVector ? 1 : length());
        for (var exp : expressions) {
            DMatrix vector = exp.evaluate(bindings);
            if (vector instanceof DMatrixRMaj v) {
                CommonOps_DDRM.add(accum, v, accum);
            } else if (vector instanceof DMatrixSparseCSC v) {
                Iterator<DMatrixSparse.CoordinateRealValue> coords = v.createCoordinateIterator();
                while (coords.hasNext()) {
                    DMatrixSparse.CoordinateRealValue coord = coords.next();
                    accum.add(coord.row, coord.col, coord.value);
                }
            } else {
                throw new UnsupportedOperationException("Cannot sum vector type " + vector.getClass());
            }
        }

        return accum;
    }

    @Override
    public DMatrix computeDerivative(Model.ParameterBindings bindings) {
        return MatrixSum.sum(Arrays.stream(expressions)
                                   .map(exp -> exp.computeDerivative(bindings))
                                   .map(DMatrixExpression::new)
                                   .toArray(MatrixExpression[]::new))
                        .evaluate(bindings);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        return VectorSum.sum(
                Arrays.stream(expressions)
                      .map(exp -> exp.computePartialDerivative(bindings, variable))
                      .map(DMatrixColumnVectorExpression::new)
                      .toArray(VectorExpression[]::new)
        ).evaluate(bindings);
    }
}
