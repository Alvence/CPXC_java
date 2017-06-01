package com.yunzhejia.unimelb.cpexpl.truth;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class LRTruth {

	public static void main(String[] args) {
		
		try {
			Instances train = DataUtils.load("data/icdm2017Data/iris_train.arff");
			Instances test = DataUtils.load("data/icdm2017Data/iris_test.arff");
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.LOGISTIC);
			cl.buildClassifier(train);
			System.out.println(cl);
			
			Instance instance = test.get(1);
			System.out.println("Ins: "+instance);
			
			getGoldFeature(cl,instance);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

	}

	public static Set<Integer> getGoldFeature(AbstractClassifier cl, Instance instance) throws Exception {
		if (!(cl instanceof Logistic)){
			System.err.println("not Logstic");
			return null;
		}
		Logistic logit = (Logistic)cl;
		
		int pred = (int)cl.classifyInstance(instance);
		Set<Integer> expl = new HashSet<>();
		for(double[] par:logit.getM_Par()){
			System.out.println((Arrays.toString(par)));
		}
		return expl;
	}
	
	
	
	private static boolean isImportant(ClassifierTree root, Instance instance){

		
		return false;
	}
}
