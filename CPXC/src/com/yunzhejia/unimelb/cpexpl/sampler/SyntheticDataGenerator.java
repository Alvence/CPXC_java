package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.datavis.ScatterPlotDemo3;

import weka.core.Instance;
import weka.core.Instances;

public class SyntheticDataGenerator {

	public static int getLabel(double x, double y, int region){
		double centerX = 5, centerY = 5;
		double r1 = 4,r2 = 1;
		
		double dis = Math.sqrt((x - centerX)*(x - centerX)+(y-centerY)*(y-centerY));
		if (dis>=r2 && dis<= r1)
			return region == 1? 1:0;
		else
			return region == 1? 0:1;
	}
	
	public static void generateData(){
		 int[] regions = {1,2};
		 Random random = new Random(1);
		 for(int n = 0; n < 2000; n++){
			 int region = 0; 
			 if (random.nextDouble()>0.5){
				 region = 1;
			 }else{
				 region = 2;
			 }
			 double x = (int)(random.nextDouble()*1000)/100.0;
			 double y =(int)(random.nextDouble()*1000)/100.0;
			 int label = getLabel(x, y, region);
			 System.out.println(region+","+x+","+y+","+label);
		 }
	}
	
	public static void main(String[] args) {
		/*try {
			Instances data = DataUtils.load("data/synthetic3.arff");
			List<Instances> datas = new ArrayList<>();
	    	for(int i = 0; i < data.numClasses();i++){
	    		datas.add(new Instances(data,0));
	    	}
	    	
	    	for (Instance ins: data){
	    		if(ins.stringValue(0).equals("2"))
	    			continue;
	    		int index = (int)ins.classValue();
	    		datas.get(index).add(ins);
	    	}
	    	
	    	ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(datas, 1, 2));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
	}

}
