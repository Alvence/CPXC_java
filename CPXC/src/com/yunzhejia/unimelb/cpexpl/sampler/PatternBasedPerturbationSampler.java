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

public class PatternBasedPerturbationSampler implements Sampler {
	PatternSet ps = null;
	Random rand = new Random(0);
	
	public PatternBasedPerturbationSampler(PatternSet patternset){
		this.ps = patternset;
	}
	@Override
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N) {
		List<Integer> slots = new ArrayList<>();
		PatternSet mp = ps.getMatchingPatterns(instance);
		int count = 0;
		for(IPattern p:mp){
			for(int i = 0 ;i < p.support(headerInfo)*100;i++){
				slots.add(count);
			}
			count++;
		}
		
		Instances ret = new Instances(headerInfo,0);
		
//		System.out.println(mp.size()+" out of "+ps.size());
		while(ret.numInstances()<N){
			int index = rand.nextInt(slots.size());
			IPattern p = mp.get(slots.get(index));
			Instance sample = perturb(headerInfo, instance, p);
			if(mp.match(sample)){
				ret.add(sample);
			}
		}
		return ret;
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
