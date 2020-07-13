package neuralnerdwork.weight;

public class VariableWeightInitializer {
    public static double dumbRandomWeightInitializer(int cols, int rows) {
        return (Math.random() - 0.5) * 2.0;
    } 

    public static double smartRandomWeightInitializer(int cols, int rows) {
        return (Math.random() - 0.5) * 2.0;
    } 

}