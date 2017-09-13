package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class GradientBasedSampler implements Sampler {
	private double delta = 0.5;
	private double eps = 0.01;
	
	public GradientBasedSampler(){
		
	}
	public GradientBasedSampler(double delta, double eps){
		this.delta = delta;
		this.eps = eps;
	}
	
	@Override
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Instances samplingFromInstance(AbstractClassifier cl, Instances headerInfo, Instance instance, int N) {
		Instances ret = new Instances(headerInfo, 0);
		int count = 0;
		Instance newIns = (Instance)instance.copy();
		Instance cur = null;
		while(count<N && (cur==null || distance(cur,newIns) > 1e-6)){
			try {
				cur = (Instance)newIns.copy();
				List<Double> gr = calcGradient(cl, headerInfo, cur);
				//normalize the gradient
				double sum = 0;
				for(double g:gr){
					sum=sum+ g*g;
				}
				sum = Math.sqrt(sum);
				if(sum!=0){
				for (int i = 0; i<gr.size();i++){
					double old = gr.get(i);
					gr.set(i, old/sum);
				}}
				
				
				newIns = (Instance)cur.copy();
				for(int i=0; i < gr.size(); i++){
					if(gr.get(i)!=0){
//						System.out.println(gr);
						double newVal = cur.value(i) - eps*(gr.get(i) + 
								(Math.random()-0.5)/10)* (headerInfo.attributeStats(i).numericStats.max-headerInfo.attributeStats(i).numericStats.min>0?(headerInfo.attributeStats(i).numericStats.max-headerInfo.attributeStats(i).numericStats.min):1);
						newIns.setValue(i, newVal);
					}
				}
				ret.add(newIns);
				count++;
//				System.out.println("count="+count+"   "+gr +"   dis=" +  distance(cur,newIns) );
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return ret;
	}
	
	public double distance(Instance x, Instance y){
		double sum = 0;
		for (int i = 0; i < x.numAttributes();i++){
			if(x.attribute(i).isNumeric()){
				sum = sum+ (x.value(i)-y.value(i))*(x.value(i)-y.value(i));
			}
		}
		return Math.sqrt(sum);
	}
	
	
	public List<Double> calcGradient(AbstractClassifier cl, Instances headerInfo, Instance instance) throws Exception{
		List<Double> ret = new ArrayList<>();
		for(int i = 0; i < instance.numAttributes();i++){
			if(!instance.attribute(i).isNumeric() && instance.classIndex()==i){
				ret.add(0.0);
			}else{
				Instance temp = (Instance)instance.copy();
				double deltaX = delta * (headerInfo.attributeStats(i).numericStats.max-headerInfo.attributeStats(i).numericStats.min>0?(headerInfo.attributeStats(i).numericStats.max-headerInfo.attributeStats(i).numericStats.min):1);
				double newVal = instance.value(i) + deltaX;
				temp.setValue(i, newVal);
				int y = (int)cl.classifyInstance(instance);
				double deltaY = cl.distributionForInstance(temp)[y] - cl.distributionForInstance(instance)[y];
				double gr = deltaY/deltaX;
				ret.add(gr);
//				System.out.println(ret+" "+deltaY+" "+deltaX);
			}
		}
		
		return ret;
		
	}

}
