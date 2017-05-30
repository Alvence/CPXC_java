package com.yunzhejia.unimelb.cpexpl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.CPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.FPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.PatternSortingStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.SamplingStrategy;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Discretize;

public class GreedyExplainer {
	private int getBin(double[] points, double value){
		int index = 0;
		while( index < points.length && value > points[index]){
			index++;
		}
		return index;
	}
	
	public List<IPattern> getExplanations(FPStrategy fpStrategy, SamplingStrategy samplingStrategy, CPStrategy cpStrategy, PatternSortingStrategy patternSortingStrategy,
			AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K, boolean debug) throws Exception{
		Discretize dis = new Discretize();
		dis.setBins(10);
		dis.setInputFormat(headerInfo);
		weka.filters.supervised.attribute.Discretize.useFilter(headerInfo, dis);
		
		
		List<IPattern> patterns = new ArrayList<>();
		for(int k = 0; k < K; k++){

			Set<ICondition> conds = new HashSet<>();
			for(int i = 0; i < headerInfo.numAttributes(); i++){
				if(reject(i,instance,cl)){
					continue;
				}
				ICondition cond = getCondition(instance, i, dis);
				conds.add(cond);
			}
			if(conds.size()!=0)
				patterns.add(new Pattern(conds));
		}
		return patterns;
	}

	private boolean reject(int i, Instance instance, AbstractClassifier cl) throws Exception {
		double pred = cl.classifyInstance(instance);
		Instance ins = (Instance)instance.copy();
		ins.setMissing(i);
		if(cl.classifyInstance(ins)!=pred)
			return false;
		else
			return true;
	}

	private ICondition getCondition(Instance instance, int attrIndex, Discretize dis) {
		if(instance.attribute(attrIndex).isNumeric()){
			double value = instance.value(attrIndex);
			int bin = getBin(dis.getCutPoints(attrIndex), value);
			double left = bin==0? Double.MIN_VALUE:dis.getCutPoints(attrIndex)[bin-1];
			double right = bin==dis.getCutPoints(attrIndex).length? Double.MAX_VALUE:dis.getCutPoints(attrIndex)[bin];
			return new NumericCondition(instance.attribute(attrIndex).name(), attrIndex, left, right);
		}else{
			return new NominalCondition(instance.attribute(attrIndex).name(), attrIndex, instance.stringValue(attrIndex));
		}
	}
}
