package neuralnerdwork;

import neuralnerdwork.math.ConstantVector;

public record TrainingSample(ConstantVector input, ConstantVector output){

    public TrainingSample(double[] input, double[] output) {
        this(new ConstantVector(input), new ConstantVector(output));
    }

}
