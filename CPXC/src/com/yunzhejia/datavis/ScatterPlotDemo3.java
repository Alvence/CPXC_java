package com.yunzhejia.datavis;

import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Instance;
import weka.core.Instances;   
   
/**  
 * A demo scatter plot with some code showing how to convert between Java2D   
 * coordinates and (x, y) coordinates.  
 */   
public class ScatterPlotDemo3 extends ApplicationFrame {         
    /**  
     * A demonstration application showing a scatter plot.  
     *  
     * @param title  the frame title.  
     */   
    public ScatterPlotDemo3(String title) {   
        super(title);   
    }   
            
    private static JFreeChart createChart(Instances data, int xIndex, int yIndex) {   
    	XYSeries series = new XYSeries(0);
    	for (Instance instance:data){
    		series.add(instance.value(xIndex), instance.value(yIndex));
    	}
    	XYDataset dataset = new XYSeriesCollection(series);
    	
        JFreeChart chart = ChartFactory.createScatterPlot("Scatter Plot",   
                "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);   
    
        NumberAxis domainAxis = (NumberAxis) chart.getXYPlot().getDomainAxis();   
        domainAxis.setAutoRangeIncludesZero(false);   
        return chart;   
    }   
    
    private static JFreeChart createChart(List<Instances> datasets, int xIndex, int yIndex) {   
    	DefaultXYDataset ds = new DefaultXYDataset();
    	int count = 0;
    	for (Instances data:datasets){
    		XYSeries series = new XYSeries(count);
    		for (Instance instance:data){
        		series.add(instance.value(xIndex), instance.value(yIndex));
        	}
    		ds.addSeries(count++, series.toArray());
    	}

    	JFreeChart chart = ChartFactory.createScatterPlot("Scatter Plot",   
                "X", "Y", ds, PlotOrientation.VERTICAL, true, true, false);   
    
        NumberAxis domainAxis = (NumberAxis) chart.getXYPlot().getDomainAxis();   
        domainAxis.setAutoRangeIncludesZero(false);   
        return chart;   
    }   
    
    public static void render(JFreeChart chart){
    	JPanel  panel = createPanel(chart);
    	JFrame frame = new JFrame();
    	frame.setSize(new java.awt.Dimension(600, 400));
    	frame.setContentPane(panel);
    	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    	RefineryUtilities.centerFrameOnScreen(frame);   
    	frame.setVisible(true);
    }
   
    /**  
     * Creates a panel for the demo (used by SuperDemo.java).  
     *   
     * @return A panel.  
     */   
    public static JPanel createPanel(JFreeChart chart) { 
    	ChartPanel chartPanel = null;
	    chartPanel = new ChartPanel(chart);   
	    chartPanel.setDomainZoomable(true);   
	    chartPanel.setRangeZoomable(true);   
        return chartPanel;   
    }   
       
    /**  
     * Starting point for the demonstration application.  
     *  
     * @param args  ignored.  
     * @throws Exception 
     */   
    public static void main(String[] args) throws Exception {   
    	Instances data = DataUtils.load("data/synthetic2.arff");
    	Instances data1 = new Instances(data,0);
    	Instances data2 = new Instances(data,0);
    	for (Instance ins: data){
    		if (ins.classValue() == 0){
    			data1.add(ins);
    		}else{
    			data2.add(ins);
    		}
    	}
    	List<Instances> datas = new ArrayList<>();
    	datas.add(data1);
    	datas.add(data2);
    	ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(datas, 0, 1));;
    }   
   
}   