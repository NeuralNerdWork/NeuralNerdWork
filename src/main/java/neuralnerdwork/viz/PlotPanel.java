package neuralnerdwork.viz;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static java.util.Objects.requireNonNull;

public class PlotPanel extends JPanel {
    private final java.util.List<Shape> shapes = new ArrayList<>();
    private final Rectangle2D dataViewport = new Rectangle2D.Double(-1.0, -1.0, 2.0, 2.0);
    private volatile Collection<PointSet> pointSets = List.of();

    public PlotPanel() {
        setPreferredSize(new Dimension(500, 500));
    }

    public void addShape(Shape shape) {
        shapes.add(requireNonNull(shape));
        repaint();
    }

    public void setDataViewport(double x1, double y1, double x2, double y2) {
        dataViewport.setFrameFromDiagonal(x1, y1, x2, y2);
        repaint();
    }

    public void setDataViewport(Rectangle2D viewport) {
        dataViewport.setFrame(viewport);
        repaint();
    }

    public void updatePointSets(Collection<PointSet> pointSets) {
        this.pointSets = pointSets;
        repaint();
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        double pointSizeInView = 4.0;

        g2.clearRect(0, 0, getWidth(), getHeight());

        AffineTransform origTransform = g2.getTransform();
        AffineTransform modelToView = modelToViewTransform();
        g2.transform(modelToView);
        double pointSize = pointSizeInView / modelToView.getScaleX();

        g2.setStroke(new BasicStroke((float) (2.0 / modelToView.getScaleX())));

        g2.setColor(Color.ORANGE);
        g2.draw(dataViewport);

        g2.setColor(Color.BLUE);
        for (Shape s : shapes) {
            g2.draw(s);
        }

        for (PointSet ps : pointSets) {
            g2.setPaint(ps.paint());
            for (double[] p : ps.points()) {
                // TODO this assumes model space is 2d. Should use some fancy projection matrix for n-dimensional plotting.
                Ellipse2D.Double dot = new Ellipse2D.Double(
                        p[0] - pointSize / 2.0, p[1] - pointSize / 2.0,
                        pointSize, pointSize);
                g2.draw(dot);
            }
        }

        g2.setTransform(origTransform);
    }

    private AffineTransform modelToViewTransform() {
        double xscale = getWidth() / dataViewport.getWidth();
        double yscale = getHeight() / dataViewport.getHeight();

        // want to lock aspect, so check X scale and Y scale and choose smaller of them
        double scale = Math.min(xscale, yscale);

        AffineTransform transform = new AffineTransform();
        // now in view space. origin is at top left.
        transform.scale(scale, scale);
        // now in model space. origin is still at top left.
        transform.translate(
                getWidth() / 2.0 / scale,
                getHeight() / 2.0 / scale);
        // still in model space. model origin is centered in view.

        return transform;
    }
}
