package com.yunzhejia.cpxc;


import java.util.Random;

import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Driver {
	public static void main(String[] args){
		runSimulation("data/ILPD.arff");
		runSimulation("data/blood.arff");
		
	}
	
	public static void runSimulation(String inputfile){
		AbstractClassifier cpxc = new CPXC(ClassifierType.NAIVE_BAYES,ClassifierType.DECISION_TREE, 0.9, 0.02,2);
		DataSource source;
		Instances data;
		try {
			source = new DataSource(inputfile);
			//source = new DataSource("data/diabetes.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
//			weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
//			discretizer.setInputFormat(data);
//			data = weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
			
			Evaluation eval = new Evaluation(data);
//			cpxc.buildClassifier(data);
//			eval.evaluateModel(cpxc, data);
			eval.crossValidateModel(cpxc, data, 7, new Random(1));
			
			System.out.println("accuracy of CPXC: " + eval.pctCorrect() + "%");
			System.out.println("AUC of CPXC: " + eval.weightedAreaUnderROC());
			
			
			/*
			AbstractClassifier cl = new NaiveBayes();
			Evaluation eval1 = new Evaluation(data);
			eval1.crossValidateModel(cl, data, 7, new Random(1));
			System.out.println("accuracy of NBC: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of NBC: " + eval1.weightedAreaUnderROC());
			*/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
