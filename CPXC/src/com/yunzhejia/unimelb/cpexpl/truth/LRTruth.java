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
			Instances train = DataUtils.load("data/icdm2017Data/diabetes_train.arff");
			Instances test = DataUtils.load("data/icdm2017Data/diabetes_test.arff");
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.LOGISTIC);
			cl.buildClassifier(train);
			System.out.println(cl);
			
			Instance instance = test.get(1);
			System.out.println("Ins: "+instance);
			
			System.out.println(getGoldFeature(cl,instance));
			
			
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
		double[] cof = new double[instance.numAttributes()-1];
		int count = 0;
		for(int i = 1; i < logit.getM_Par().length;i++){
			double[] par = logit.getM_Par()[i];
//			System.out.println((Arrays.toString(par)));
			cof[count++] = par[0];
		}
//		System.out.println((Arrays.toString(cof)));
		if(pred == 1){
			double sum = 0;
			for(int i = 0; i < cof.length;i++){
				if (cof[i]*instance.value(i)>0){
					sum+= cof[i]*instance.value(i);
				}
			}
			for(int i = 0; i < cof.length;i++){
				if (cof[i]*instance.value(i)/sum > 0.1){
					expl.add(i);
				}
			}
		}else{
			double sum = 0;
			for(int i = 0; i < cof.length;i++){
				if (cof[i]*instance.value(i)<0){
					sum+= cof[i]*instance.value(i);
				}
			}
			for(int i = 0; i < cof.length;i++){
				if (cof[i]*instance.value(i)/sum > 0.1){
					expl.add(i);
				}
			}
		}
		return expl;
	}
	
	
	
	private static boolean isImportant(ClassifierTree root, Instance instance){

		
		return false;
	}
}
