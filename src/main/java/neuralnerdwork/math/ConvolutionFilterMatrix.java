package neuralnerdwork.math;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A matrix capable of representing a convolution filter (without padding) as a linear transformation
 * (i.e. multiplying a flattened vector by this filter is equivalent to applying the given
 * convolution filter over the un-flattened source array).
 *
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
    public Matrix evaluate(Model.ParameterBindings bindings) {
        Matrix filter = this.filter.evaluate(bindings);
        Map<SparseConstantMatrix.Index, Double> values = new LinkedHashMap<>();

        int targetCols = inputWidth - this.filter.cols() + 1;
        int targetRows = inputHeight - this.filter.rows() + 1;
        int flattenedLength = targetCols * targetRows;

        int sourceCol = 0, paddingCounter = 0;
        for (int flattenedRow = 0; flattenedRow < flattenedLength; flattenedRow++) {
            int startIndex = sourceCol;
            for (int r = 0, i = 0; r < filter.rows(); r++) {
                for (int c = 0; c < filter.cols(); c++, i++) {
                    values.put(new SparseConstantMatrix.Index(flattenedRow, startIndex + i), filter.get(r, c));
                }
                i += (inputWidth - filter.cols());
            }
            paddingCounter++;
            if (paddingCounter == filter.cols()) {
                sourceCol += filter.cols();
                paddingCounter = 0;
            } else {
                sourceCol++;
            }
        }

        return new SparseConstantMatrix(values, rows(), cols());
    }

    @Override
    public MatrixExpression computePartialDerivative(int variable) {
        return null;
    }
}
