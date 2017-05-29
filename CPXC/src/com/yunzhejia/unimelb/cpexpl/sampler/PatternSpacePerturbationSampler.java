package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instance;
import weka.core.Instances;

public class PatternSpacePerturbationSampler implements Sampler {
	PatternSet ps = null;
	Random rand = new Random(0);
	double threshold = 3;
	
	public PatternSpacePerturbationSampler(PatternSet patternset, double threshold){
		this.ps = patternset;
		this.threshold = threshold;
	}
	
	private double l1dis(double[]xp, double[]yp){
		double ret = 0.0;
		for(int i = 0; i < xp.length; i++){
			ret += Math.abs(xp[i]-yp[i]);
		}
		return ret;
	}
	
	public double distance(Instance x, Instance y, Instances headerInfo){
		double[] xp = new double[ps.size()];
		double[] yp = new double[ps.size()];
		
		for(int i = 0; i < ps.size(); i++){
			IPattern p = ps.get(i);
			if(p.match(x)){
				xp[i] = p.support(headerInfo);
			}else{
				xp[i] = 0;
			}
			if(p.match(y)){
				yp[i] = p.support(headerInfo);
			}else{
				yp[i] = 0;
			}
		}
		
		return l1dis(xp,yp);
	}
	
	@Override
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N) {
		Instances samples = new Instances(headerInfo, 0);
		while(samples.size()<N){
			Instance sample = perturb(headerInfo,instance);
			if(distance(sample,instance, headerInfo) < threshold){
				samples.add(sample);
			}
		}
		return samples;
	}
	
	private Instance perturb(Instances data, Instance instance){
		//uniformly select number of features to be perturbed.
		
		Instance newIns = (Instance)instance.copy();
		
		for (int i = 0 ; i < instance.numAttributes(); i++){
			if (rand.nextDouble()>0.5 || i == data.classIndex()){
				continue;
			}
			else{
				if (data.attribute(i).isNumeric()){
					double newValue =  perturbNumericValue(data,instance,i);
					newIns.setValue(i, newValue);
				}else{
					String newValue =  perturbNonimalValue(data,instance,i);
					newIns.setValue(i, newValue);
				}
			}
		}
		return newIns;
	}
	
	private Instance perturb(Instances data, Instance instance, IPattern p) {
		Instance newIns = (Instance)instance.copy();
		
		for (int i = 0 ; i < instance.numAttributes(); i++){
			ICondition condition = appear(i,p);
			if (condition!=null){
				if (data.attribute(i).isNumeric()){
					NumericCondition numCond = (NumericCondition)condition;
					double left = numCond.getLeft();
					double right = numCond.getRight();
					if (left == Double.MIN_VALUE){
						left = data.attributeStats(i).numericStats.min;
					}
					if (right == Double.MAX_VALUE){
						right = data.attributeStats(i).numericStats.max;
					}
					double newValue = left+(right-left)*rand.nextDouble();
					newIns.setValue(i, newValue);
				}
			}
			
			if (rand.nextDouble()>0.5 || i == data.classIndex()){
				continue;
			}
			else{
				if (data.attribute(i).isNumeric()){
					double newValue =  perturbNumericValue(data,instance,i);
					newIns.setValue(i, newValue);
				}else{
					String newValue =  perturbNonimalValue(data,instance,i);
					newIns.setValue(i, newValue);
				}
			}
		}
		return newIns;
	}
	
	private ICondition appear(int i, IPattern p) {
		for(ICondition cond:p.getConditions()){
			if (cond.getAttrIndex()==i){
				return cond;
			}
		}
		return null;
	}
	private String perturbNonimalValue(Instances data, Instance instance, int attrIndex) {
		Enumeration<Object> strs = data.attribute(attrIndex).enumerateValues();
		List<String> values = new ArrayList<>();
		while(strs.hasMoreElements()){
			values.add((String)strs.nextElement());
		}
		int index = rand.nextInt(values.size());
		return values.get(index);
	}

	private double perturbNumericValue(Instances data, Instance instance, int attrIndex) {
		double max = data.attributeStats(attrIndex).numericStats.max;
		double min = data.attributeStats(attrIndex).numericStats.min;
		double std = data.attributeStats(attrIndex).numericStats.stdDev;
		double mean = data.attributeStats(attrIndex).numericStats.mean;
		double unit = data.attributeStats(attrIndex).numericStats.max / data.attributeStats(attrIndex).numericStats.mean;
		double val = (instance.isMissing(attrIndex)?mean:instance.value(attrIndex))+ rand.nextGaussian()*std;
		val = ((int)(val*100))/100.00;
		return val;
//		return (rand.nextDouble() * (max - min)) + min;
	}

}
