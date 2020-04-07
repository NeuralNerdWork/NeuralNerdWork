package neuralnerdwork.math;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record ParameterMatrix(String variablePrefix, int rows, int cols) implements MatrixFunction {
    @Override
    public Matrix apply(ScalarVariableBinding[] argument) {
        final double[][] values = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                final MatrixIndexVariable indexVariable = new MatrixIndexVariable(variablePrefix, i, j);
                final ScalarVariableBinding scalarVariableBinding = Arrays.stream(argument)
                                                                          .filter(binding -> binding.variable().equals(indexVariable))
                                                                          .findFirst()
                                                                          .orElseThrow(() -> new IllegalStateException(String.format("Variable %s is not valid for [%s]", indexVariable.symbol(), this)));
                values[i][j] = scalarVariableBinding.value();
            }
        }

        return new ConstantMatrix(values);
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
