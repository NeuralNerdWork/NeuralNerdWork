package neuralnerdwork.math;

import java.util.Arrays;

public record ScalarComponentsFunction(ScalarFunction[] components) implements VectorFunction {
    public ScalarComponentsFunction {
        if (components == null || components.length == 0) {
            throw new IllegalArgumentException("Cannot have null or empty comopnents");
        }
    }

    @Override
    public int length() {
        return components.length;
    }

    @Override
    public Vector apply(double[] inputs) {
        return new ConstantVector(
                Arrays.stream(components)
                      .mapToDouble(f -> f.apply(inputs))
                      .toArray()
        );
    }

    @Override
    public VectorFunction differentiate(int variableIndex) {
        return new ScalarComponentsFunction(
                Arrays.stream(components)
                      .map(f -> f.differentiate(variableIndex))
                      .toArray(ScalarFunction[]::new)
        );
    }

    @Override
    public int inputLength() {
        return Arrays.stream(components)
                     .mapToInt(ScalarFunction::inputLength)
                     .max()
                     .orElseThrow();
    }
}
