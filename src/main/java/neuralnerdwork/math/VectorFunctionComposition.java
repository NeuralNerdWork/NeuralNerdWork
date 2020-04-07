package neuralnerdwork.math;

import java.util.Arrays;

public record VectorFunctionComposition(VectorFunction... functions) implements VectorFunction {
    public VectorFunctionComposition {
        if (functions == null || functions.length == 0) {
            throw new IllegalArgumentException("Cannot compose null or empty functions");
        }
    }

    @Override
    public int inputLength() {
        return functions[0].inputLength();
    }

    @Override
    public int length() {
        return functions[functions.length-1].length();
    }

    @Override
    public Vector apply(double[] inputs) {
        Vector cur = new ConstantVector(inputs);
        for (VectorFunction f : functions) {
            cur = f.apply(cur.toArray());
        }

        return cur;
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        if (functions.length > 1) {
            // Chain rule works from outside function to inside
            final int last = functions.length-1;
            final MatrixFunction lastDerivative = functions[last].differentiate();

            // Chain Rule
            // (fog)'(x) = f'(g(x))*g'(x)
            final VectorFunction innerFunction = last != 1 ? new VectorFunctionComposition(Arrays.copyOf(this.functions, last))
                    : this.functions[0];
            return new MatrixVectorProductFunction(
                    new MatrixCompose(innerFunction, lastDerivative),
                    innerFunction.differentiate(variableIndex)
            );
        } else {
            return functions[0].differentiate(variableIndex);
        }
    }
}
