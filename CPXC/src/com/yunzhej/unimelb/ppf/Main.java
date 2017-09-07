package com.yunzhej.unimelb.ppf;

import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
	
	public static void main(String[] args) {
//		String[] files = {"iris.arff","blood.arff",
//				"breast-cancer.arff","hepatitis.arff","ILPD.arff",
//				"mushroom.arff","sick.arff","titanic.arff"};
		String[] files = {"iris.arff"};
		try {
		 
			PrintWriter writer = new PrintWriter(new File("tmp/results_PPF.txt"));
		for(String file:files){
			writer.println(file);
			System.out.println(file);
		
			//for(int s = 0; s < 10; s++){
		
			Instances data = DataUtils.load("data/"+file);
			data.randomize(new Random(0));
			Instances train = data.trainCV(5, 0);
			Instances test = data.testCV(5, 0);
			
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.RANDOM_FOREST);
			cl.buildClassifier(train);
			
			CostFunction.init(train.numAttributes(),0);
			
			for(int i=0; i < train.numClasses();i++){
				calculate(cl,test,0.1,i,writer);
			//}
//			featureTweeking
			}
		}
		
		writer.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void calculate(AbstractClassifier cl, Instances xs, double eps, double positiveLabel,PrintWriter writer) throws Exception{
		int total = 0;
		int numFlipped = 0;
		double cost = 0;
		PPF featureTweeking = new PPF();
		int i = 0 ;
		for (Instance x:xs){
//			if(i++!=3){
//				continue;
//			}
			if (x.classValue() != positiveLabel){
				total++;
				Instance y = featureTweeking.featureTweaking(cl, x, xs, eps, positiveLabel);
				if (y!=null){
					numFlipped++;
					cost += CostFunction.cost(x, y);
				}
			}
		}
		writer.println("Total = "+total+" Feasible="+numFlipped+" Feasible solution = "+numFlipped*1.0/total*100+"%    cost="+cost);
		System.out.println("Total = "+total+" Feasible="+numFlipped+" Feasible solution = "+numFlipped*1.0/total*100+"%    cost="+cost);
	}

}
