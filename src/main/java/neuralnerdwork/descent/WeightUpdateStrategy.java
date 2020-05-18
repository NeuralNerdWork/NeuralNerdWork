package neuralnerdwork.descent;

import neuralnerdwork.math.Vector;

public interface WeightUpdateStrategy {
    Vector updateVector(Vector rawGradient);
}
