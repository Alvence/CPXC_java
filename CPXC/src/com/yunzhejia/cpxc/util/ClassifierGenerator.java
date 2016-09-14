package com.yunzhejia.cpxc.util;

import java.io.Serializable;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;

public class ClassifierGenerator implements Serializable{
	private static final long serialVersionUID = -2578826239599092125L;

	public enum ClassifierType {NAIVE_BAYES, DECISION_TREE, SVM, LOGISTIC};
	
	public static AbstractClassifier getClassifier(ClassifierType type){
		AbstractClassifier classifier = null;
		switch(type){
			case NAIVE_BAYES:
				classifier = new NaiveBayes();
				break;
			case DECISION_TREE:
				classifier = new J48();
				break;
			case SVM:
				classifier = new LibSVM();
				break;
			case LOGISTIC:
				classifier = new Logistic();
				break;
			default:
				classifier = new NaiveBayes();
				break;
				
		}
		return classifier;
	}
}
