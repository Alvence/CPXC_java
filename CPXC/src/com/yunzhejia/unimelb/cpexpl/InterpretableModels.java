package com.yunzhejia.unimelb.cpexpl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;

public class InterpretableModels {
	
	public static Set<Integer> getGoldenFeature(ClassifierType type, AbstractClassifier cl, Instances data){
		Set<Integer> ret = new HashSet<>();
		List<Double> weights = new ArrayList<>();
		double sum = 0;
		switch(type){
		case LOGISTIC:
			Logistic logistic = (Logistic)cl;
			for(int i = 1; i < logistic.getM_Par().length;i++){
				double[] par = logistic.getM_Par()[i];
				weights.add(par[0]);
				sum+=par[0];
			}
			break;
		default:
			break;
		}
		for(int i = 0;i<weights.size();i++){
			if (weights.get(i)*100/sum > 5){
				ret.add(i);
			}
		}
//		System.out.println(ret);
		return ret;
	}

	public static void main(String[] args) {
		try {
			Instances data = DataUtils.load("data/balloon.arff");
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
			cl.buildClassifier(data);
			getGoldenFeature(ClassifierType.LOGISTIC, cl, data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
