package com.yunzhejia.datavis;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Instances;


public class VisualizeData extends JFrame {
    public VisualizeData(Instances data) {

        initUI(data);
    }

    private void initUI(Instances data) {

        final Surface surface = new Surface(data);
        add(surface);
        
        setTitle("Points");
        setSize(350, 250);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static void draw(Instances data) {

            VisualizeData ex = new VisualizeData(data);
            ex.setVisible(true);
        
    }
    
    public static void main(String[] args) throws Exception {
    	Instances data = DataUtils.load("data/synthetic2.arff");
    	draw(data);
        
    }
    
    class Surface extends JPanel implements ActionListener {
    	public Surface(Instances data){
    		super();
    		this.data=data;
    	}
    	Instances data;
        private void doDrawing(Graphics g) {

            Graphics2D g2d = (Graphics2D) g;

            g2d.setPaint(Color.blue);
            
            int w = getWidth();
            int h = getHeight();
            
            int ra =2;

            for (int i = 0; i < data.numInstances(); i++) {

                int x = ( (int)Math.abs(data.get(i).value(0))) % w;
                int y = ( (int)Math.abs(data.get(i).value(1))) % h;
//                g2d.drawLine(x, y, x, y);
                g2d.drawOval( x - ra/2, y - ra/2, ra, ra );
            }
        }

        @Override
        public void paintComponent(Graphics g) {

            super.paintComponent(g);
            doDrawing(g);
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            repaint();
        }
    }

}