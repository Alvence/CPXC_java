package com.yunzhejia.cpxc;

import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class LocalClassifier {
	private Pattern pattern;
	private AbstractClassifier classifier;
	private double weight;
	
	public LocalClassifier(Pattern pattern, ClassifierType type){
		this.pattern = pattern;
		this.classifier = ClassifierGenerator.getClassifier(type);
		this.weight = 1;
	}
	
	public AbstractClassifier getClassifier(){
		return classifier;
	}
	
	public void train(Instances data) throws Exception{
		classifier.buildClassifier(data);
	}
	
	public double predict(Instance ins) throws Exception{
		return classifier.classifyInstance(ins);
	}
	
	public double getWeight(){
		return weight;
	}
	
}
