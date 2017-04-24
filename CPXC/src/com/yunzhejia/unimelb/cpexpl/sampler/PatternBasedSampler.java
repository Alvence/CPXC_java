package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import com.yunzhejia.pattern.PatternSet;

import weka.core.Instance;
import weka.core.Instances;

public class PatternBasedSampler implements Sampler {
	PatternSet ps = null;
	Random rand = new Random(0);
	
	public PatternBasedSampler(PatternSet ps){
		this.ps = ps;
	}

	@Override
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N) {
		Instances ret = new Instances(headerInfo,0);
		PatternSet mp = ps.getMatchingPatterns(instance);
//		System.out.println(mp.size()+" out of "+ps.size());
		while(ret.numInstances()<N){
			Instance sample = perturb(headerInfo, instance);
			if(mp.match(sample)){
				ret.add(sample);
			}
		}
		return ret;
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
