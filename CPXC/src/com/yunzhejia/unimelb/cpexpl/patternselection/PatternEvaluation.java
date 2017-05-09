package com.yunzhejia.unimelb.cpexpl.patternselection;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class PatternEvaluation {
	Random rand = new Random(0);
	private double prediction(AbstractClassifier cl, Instance instance, double classIndex) throws Exception{
		return  cl.distributionForInstance(instance)[(int)classIndex];
	}
	
	private double getRand(double lower, double upper){
		return lower + rand.nextDouble()*(upper-lower);
	}
	public double predictionByRemovingPattern(AbstractClassifier cl, Instance instance, IPattern pattern, Instances headerInfo) throws Exception{
		
		Instance ins = (Instance)instance.copy();
		
		List<List<String>> values = new ArrayList<>();
		for(int i = 0; i < instance.numAttributes();i++){
			values.add(new ArrayList<String>());
		}
		
		int numNumericAttr = 5;
		
		for (ICondition condition:pattern.getConditions()){
			if (condition instanceof NominalCondition){
				String val = ((NominalCondition) condition).getValue();
				Enumeration<Object> enums = headerInfo.attribute(condition.getAttrIndex()).enumerateValues();
				while(enums.hasMoreElements()){
					String o = (String)enums.nextElement();
					if(!o.equals(val)){
						values.get(condition.getAttrIndex()).add(o);
					}
				}
			}else{
				double left = ((NumericCondition)condition).getLeft();
				double right = ((NumericCondition)condition).getRight();
				if(left!=Double.MIN_VALUE){
					double upper = left;
					double lower = headerInfo.attributeStats(condition.getAttrIndex()).numericStats.min;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
				if(right!=Double.MAX_VALUE){
					double upper = headerInfo.attributeStats(condition.getAttrIndex()).numericStats.max;
					double lower = right;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
			}
		}
		for(int i = 0; i < values.size();i++){
			if(values.get(i).size()>0){
				String val = values.get(i).get(rand.nextInt(values.get(i).size()));
				if(ins.attribute(i).isNumeric()){
					ins.setValue(i, Double.parseDouble(val));
				}else{
					ins.setValue(i, val);
				}
			}
		}
		/*
		Instances tmp = new Instances(data,0);
		int[] caps = new int[values.size()];
		int[] curs = new int[values.size()];
		for(int i =0; i < values.size();i++){
			caps[i] = (values.get(i).size());
			curs[i] = 0;
		}
		
		int pos = 0;
		while(true){
			if (pos == values.size()){
				break;
			}
			if(curs[pos] == caps[pos]){
				curs[pos] = 0;
				pos++;
			}
		}*/
		
		
//		System.out.println(ins);
		int classIndex = (int)cl.classifyInstance(instance);
		
		return prediction(cl,ins,classIndex);
	}	

}
