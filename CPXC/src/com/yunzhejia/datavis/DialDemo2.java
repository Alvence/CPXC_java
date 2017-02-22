package com.yunzhejia.datavis;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.general.DefaultValueDataset;
import org.jfree.data.general.Series;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYSeries;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class DialDemo2 extends JFrame
{
        static class DemoPanel extends JPanel
                implements ChangeListener
        {
        	
        	
        		Instances data;
        		XYSeries s1;
        		XYSeries s2;
                JPanel chartpanel;
                JSlider slider1;
                JSlider slider2;
                double thresh;
                AbstractClassifier globalCL;
                List<Double> errs;
                public void stateChanged(ChangeEvent changeevent)
                {
                		thresh = slider1.getValue()*1.0/1000;
                		resetSeries(thresh);
                        remove(chartpanel);
                        chartpanel = createPanel(createChart());
                        chartpanel.setPreferredSize(new Dimension(400, 400));
                        add(chartpanel);
                        revalidate();
                        repaint();
                }
                
                public JPanel createPanel(JFreeChart chart) { 
                	ChartPanel chartPanel = null;
            	    chartPanel = new ChartPanel(chart);   
            	    chartPanel.setDomainZoomable(true);   
            	    chartPanel.setRangeZoomable(true);   
                    return chartPanel;   
                }   

                private void resetSeries(double v){
                	s1 = new XYSeries("LE");
                	s2 = new XYSeries("SE");
                	System.out.println("threshold = "+v);
                	
            		//get cutting point
            		double k = v;
            		//initialize two data sets
            		for (int i = 0; i < data.numInstances(); i++){
            			Instance instance = data.get(i);
            			if (errs.get(i) > k){
            				s1.add(instance.value(0), instance.value(1));
            			}else{
            				s2.add(instance.value(0), instance.value(1));
            			}
            		}
					
                }
                
                private  JFreeChart createChart() {   
                	
                	DefaultXYDataset dataset = new DefaultXYDataset();
                	dataset.addSeries("LE", s1.toArray());
                	dataset.addSeries("SE", s2.toArray());
                	
                    JFreeChart chart = ChartFactory.createScatterPlot("Scatter Plot",   
                            "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);   
                
                    NumberAxis domainAxis = (NumberAxis) chart.getXYPlot().getDomainAxis();   
                    domainAxis.setAutoRangeIncludesZero(false);   
                    return chart;   
                }   
                public DemoPanel(Instances data)
                {
                	this.data = data;
                	globalCL = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
                	
            		errs = new ArrayList<Double>();
            		double[][] dist;
				
						try {
							globalCL.buildClassifier(data);
							dist = globalCL.distributionsForInstances(data);
							for (int i = 0; i < data.numInstances(); i++){
		            			Instance ins = data.get(i);
		            			int label = (int)ins.classValue();
		            			errs.add(1-dist[i][label]);
		            		}
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					
                	setLayout(new BorderLayout());
                	resetSeries(500);
                        //ChartFactory.create
                        
                        JFreeChart jfreechart = createChart();
                        jfreechart.setTitle("Dial Demo 2");
                        chartpanel = createPanel(jfreechart);
                        chartpanel.setPreferredSize(new Dimension(400, 400));
                        JPanel jpanel = new JPanel(new GridLayout(2, 1));
                        jpanel.add(new JLabel("threshold:"));
                        slider1 = new JSlider(0, 1000);
                        slider1.setMajorTickSpacing(20);
                        slider1.setPaintTicks(true);
                        slider1.setPaintLabels(true);
                        slider1.addChangeListener(this);
                        jpanel.add(slider1);
                        add(jpanel, "South");
                        add(chartpanel,"North");
                }
                
                
                
                private void divideData(Instances data, Instances LE, Instances SE) throws Exception{
                	
            	}
        }


        public DialDemo2(Instances data)
        {
                setDefaultCloseOperation(3);
                setContentPane(createDemoPanel(data));
        }

        public static JPanel createDemoPanel(Instances data)
        {
                return new DemoPanel(data);
        }

        public static void main(String args[]) throws Exception
        {
        		Instances data = DataUtils.load("data/banana.arff");
                DialDemo2 dialdemo2 = new DialDemo2(data);
                dialdemo2.pack();
                dialdemo2.setVisible(true);
        }
}