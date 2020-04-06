package neuralnerdwork.math;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record ParameterMatrix(String variablePrefix, int rows, int cols) implements MatrixFunction {
    @Override
    public Matrix apply(VectorVariableBinding argument) {
        final double[][] values = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                final String indexSymbol = new MatrixIndexVariable(variablePrefix, i, j).symbol();
                int valueIndex = getIndexOf(indexSymbol, argument.variable().variables());
                values[i][j] = argument.value().get(valueIndex);
            }
        }

        return new ConstantMatrix(values);
    }

    private int getIndexOf(String indexSymbol, ScalarVariable[] variables) {
        for (int i = 0; i < variables.length; i++) {
            if (Objects.equals(indexSymbol, variables[i].symbol())) {
                return i;
            }
        }

        throw new IllegalStateException(String.format("Variable %s is not valid for [%s]", indexSymbol, this));
    }

    @Override
    public MatrixFunction differentiate(ScalarVariable variable) {
        if (variable instanceof MatrixIndexVariable matrixIndexVar) {
            if (matrixIndexVar.prefix().equals(variablePrefix)) {
                return new SparseConstantMatrix(Map.of(new SparseConstantMatrix.Index(matrixIndexVar.row(), matrixIndexVar.col()), 1.0), rows, cols);
            }
        }
        return new SparseConstantMatrix(Map.of(), rows, cols);
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.iterate(0, n -> n + 1)
                     .limit(rows * cols)
                     .map(n -> new MatrixIndexVariable(variablePrefix, n / cols, n % cols))
                     .collect(Collectors.toSet());
    }

    public ScalarVariable variableFor(int row, int col) {
        return new MatrixIndexVariable(variablePrefix, row, col);
    }

    private record MatrixIndexVariable(String prefix, int row, int col) implements ScalarVariable {
        @Override
        public String toString() {
            return symbol();
        }

        @Override
        public String symbol() {
            return prefix + "(" + row + "," + col + ")";
        }
    }
}
