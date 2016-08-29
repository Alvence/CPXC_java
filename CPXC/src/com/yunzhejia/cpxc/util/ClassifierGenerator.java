package com.yunzhejia.cpxc.util;

import java.io.Serializable;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;

public class ClassifierGenerator implements Serializable{
	private static final long serialVersionUID = -2578826239599092125L;

	public enum ClassifierType {NAIVE_BAYES};
	
	public static AbstractClassifier getClassifier(ClassifierType type){
		AbstractClassifier classifier = null;
		switch(type){
			case NAIVE_BAYES:
				classifier = new NaiveBayes();
				break;
			default:
				classifier = new NaiveBayes();
				break;
				
		}
		return classifier;
	}
}
