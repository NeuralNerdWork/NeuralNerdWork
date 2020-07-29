package neuralnerdwork.math;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DMatrixSparseTriplet;
import org.ejml.ops.ConvertDMatrixStruct;

import java.util.SortedMap;
import java.util.TreeMap;

import static java.util.Collections.emptySortedMap;

/**
 * A matrix capable of representing a convolution filter (without padding) as a linear transformation
 * (i.e. multiplying a flattened vector by this filter is equivalent to applying the given
 * convolution filter over the un-flattened source array).
 *
 * When multiplying with a vector, that vector should be the rows of a 2D array with dimensions {@link #inputWidth} and
 * {@link #inputHeight}.
 */
public record ConvolutionFilterMatrix(ParameterMatrix filter, int inputHeight, int inputWidth) implements MatrixExpression {

    @Override
    public int rows() {
        return (inputHeight - filter.rows() + 1) * (inputWidth - filter.cols() + 1);
    }

    @Override
    public int cols() {
        return inputHeight * inputWidth;
    }

    @Override
    public boolean isZero() {
        return false;
    }

    @Override
    public DMatrix evaluate(Model.ParameterBindings bindings) {
        DMatrix filter = this.filter.evaluate(bindings);
        DMatrixSparseTriplet sparseBuilder = new DMatrixSparseTriplet(rows(), cols(), filter.getNumRows() * filter.getNumCols() * rows());

        int filterRows = filter.getNumRows();
        int filterCols = filter.getNumCols();
        int targetCols = inputWidth - filterCols + 1;
        int targetRows = inputHeight - filterRows + 1;
        int flattenedLength = targetCols * targetRows;

        int sourceCol = 0, paddingCounter = 0;
        for (int flattenedRow = 0; flattenedRow < flattenedLength; flattenedRow++) {
            int startIndex = sourceCol;
            for (int r = 0, colOffset = startIndex; r < filterRows; r++, colOffset += inputWidth) {
                for (int c = 0; c < filterCols; c++) {
                    sparseBuilder.addItem(flattenedRow, colOffset + c, filter.get(r, c));
                }
            }
            paddingCounter++;
            if (paddingCounter == targetCols) {
                sourceCol += filterCols;
                paddingCounter = 0;
            } else {
                sourceCol++;
            }
        }

        return ConvertDMatrixStruct.convert(sparseBuilder, (DMatrixSparseCSC) null);
    }

    @Override
    public DMatrix computePartialDerivative(Model.ParameterBindings bindings, int variable) {
        if (filter.containsVariable(variable)) {
            int filterRow = filter.rowIndexFor(variable);
            int filterCol = filter.colIndexFor(variable);
            int flatOffset = filterRow * inputWidth + filterCol;

            DMatrixSparseTriplet sparseBuilder = new DMatrixSparseTriplet(rows(), cols(), filter.rows() * filter.cols() * rows());

            int targetCols = inputWidth - this.filter.cols() + 1;
            int targetRows = inputHeight - this.filter.rows() + 1;
            int flattenedLength = targetCols * targetRows;

            int sourceCol = 0, paddingCounter = 0;
            for (int flattenedRow = 0; flattenedRow < flattenedLength; flattenedRow++) {
                int startIndex = sourceCol;
                sparseBuilder.addItem(flattenedRow, startIndex + flatOffset, 1.0);
                paddingCounter++;
                if (paddingCounter == targetCols) {
                    sourceCol += filter.cols();
                    paddingCounter = 0;
                } else {
                    sourceCol++;
                }
            }

            return ConvertDMatrixStruct.convert(sparseBuilder, (DMatrixSparseCSC) null);
        } else {
            return new DMatrixSparseCSC(rows(), cols(), 0);
        }
    }
}
