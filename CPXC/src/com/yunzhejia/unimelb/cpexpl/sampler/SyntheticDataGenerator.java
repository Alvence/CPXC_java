package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.Random;

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
	
	public static void LOG(int N){
		Random random = new Random(1);
//		double[] coef = {1.0,1.5,-2.0,3.0,-0.5,3,-2.5,-2,1};
		double[] coef = {1, -1, 1, 0};
		for(int i = 0; i < N; i++){
			double sum=0.0;
			for(int a = 0; a< coef.length-1; a++){
				double val = (int)(random.nextDouble()*1000)/10.0;
				System.out.print(val+",");
				sum+=val*coef[a];
			}
			sum+= coef[coef.length-1];
			sum+= random.nextGaussian();
			if(sum>0){
				System.out.println("1");
			}else{
				System.out.println("0");
			}
		}
		
	}
	
	public static void rules(int N){
		Random random = new Random(1);
		int numFeature = 6;
		for(int i = 0; i < N; i++){
			boolean vals[]=new boolean[numFeature];
			for(int a = 0; a< numFeature; a++){
				boolean val = random.nextBoolean();
				vals[a] = val;
				
			}
			
			boolean result = (vals[0]&&vals[1])||(vals[2]&vals[3])||(vals[4]&vals[5]);
			if(result){
			for(int a = 0; a< numFeature; a++){
				System.out.print((vals[a]?"1":"0")+",");
			
			
				
			}
			System.out.println(result?"1":"0");
			}
		}
	}
	
	
	public static void DNF(int N){
		Random random = new Random(1);
		int numFeature = 9;
		for(int i = 0; i < N; i++){
			int region = random.nextInt(4)+1;
			boolean vals[]=new boolean[numFeature];
			for(int a = 0; a< numFeature; a++){
				boolean val = random.nextBoolean();
				vals[a] = val;
				if(a == 0)
					continue;
			}
			boolean result = false;
			switch(region){
			case 1:
				result = vals[1]&&vals[2];
				break;
			case 2:
				result = vals[3]&&vals[4];
				break;
			case 3:
				result = vals[5]&&vals[6];
				break;
			case 4:
				result = vals[7]&&vals[8];
				break;
			}
			
			if(result){
				System.out.print(region+",");
				for(int a = 0; a< numFeature; a++){
					boolean val =vals[a];
					if(a == 0)
						continue;
					System.out.print((val?"1":"0")+",");
				}
				System.out.println(result?"1":"0");
			}
		}
		
	}
	
	
	public static void DNF2G(int N){
		Random random = new Random(1);
		int numFeature = 8;
		for(int i = 0; i < N; i++){
			boolean vals[]=new boolean[numFeature];
			for(int a = 0; a< numFeature; a++){
				boolean val = random.nextBoolean();
				vals[a] = val;
				if(a == 0)
					continue;
			}
			boolean result = false;

			if(vals[0] && vals[1]){
				result = vals[2] & vals[3] & vals[4];
			}
			else if(!vals[0] && vals[1]){
				result = vals[5] & vals[6] & vals[7];
			}
			
			if(result){
				for(int a = 0; a< numFeature; a++){
					boolean val =vals[a];
					System.out.print((val?"1":"0")+",");
				}
				System.out.println(result?"1":"0");
			}
		}
		
	}
	
	
	public static void DNF3G(int N){
		Random random = new Random(199);
		int numFeature = 19;
		for(int i = 0; i < N; i++){
			boolean vals[]=new boolean[numFeature];
			for(int a = 0; a< numFeature; a++){
				boolean val = random.nextBoolean();
				vals[a] = val;
				if(a == 0)
					continue;
			}
			boolean result = false;

			if(!vals[0] && !vals[1] && !vals[2]){
				result = vals[3] & vals[4];
			} else if(!vals[0] && !vals[1] && vals[2]){
				result = vals[5] & vals[6];
			} else if(!vals[0] && vals[1] && !vals[2]){
				result = vals[7] & vals[8];
			} else if(!vals[0] && vals[1] && vals[2]){
				result = vals[9] & vals[10];
			} else if(vals[0] && !vals[1] && !vals[2]){
				result = vals[11] & vals[12];
			} else if(vals[0] && !vals[1] && vals[2]){
				result = vals[13] & vals[14];
			} else if(vals[0] && vals[1] && !vals[2]){
				result = vals[15] & vals[16];
			} else if(vals[0] && vals[1] && vals[2]){
				result = vals[17] & vals[18];
			} else {
				System.err.println("Should not reach here");
			}
			
			if(result){
				for(int a = 0; a< numFeature; a++){
					boolean val =vals[a];
					System.out.print((val?"1":"0")+",");
				}
				System.out.println(result?"1":"0");
			}
		}
		
	}
	
	public static void main(String[] args) {
//		LOG(1000);
//		DNF2G(200);
//		DNF3G(500);
		rules(100);
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
