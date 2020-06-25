package neuralnerdwork.viz;

import java.awt.*;
import java.util.Collection;

record PointSet(Paint paint, Collection<double[]>points) {} // TODO change to Collection<TrainingSample>?