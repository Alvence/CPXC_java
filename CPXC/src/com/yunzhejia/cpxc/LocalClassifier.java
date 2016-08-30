package com.yunzhejia.cpxc;

import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class LocalClassifier {
	private Pattern pattern;
	private AbstractClassifier classifier;
	private double weight;
	private Evaluation eval;
	
	public LocalClassifier(Pattern pattern, ClassifierType type){
		this.pattern = pattern;
		this.classifier = ClassifierGenerator.getClassifier(type);
		this.weight = 1;
	}
	
	public void setWeight(double weight){
		this.weight = weight;
	}
	
	public double getWeight(){
		return weight;
	}
	
	public Pattern getPattern(){
		return pattern;
	}
	
	public AbstractClassifier getClassifier(){
		return classifier;
	}
	
	public void train(Instances data) throws Exception{
		eval = new Evaluation(data);
		classifier.buildClassifier(data);
	}
	
	public double predict(Instance instance) throws Exception{
		return classifier.classifyInstance(instance);
	}
	
	public double[] distributionForInstance(Instance instance) throws Exception{
		return classifier.distributionForInstance(instance);
	}
	
	public double AER(Instances mds, AbstractClassifier baseClassifier) throws Exception{
		double AER = 0;
		double sum = 0;
		for (Instance ins:mds){
			double errB = getError(baseClassifier,ins);
			double errP = getError(ins);
			AER += Math.abs(errB - errP);
			sum += errB;
		}
		AER = AER / sum;
		return AER;
	}
	
	public double getError(Instance ins) throws Exception{
		return getError(classifier,ins);
	}
	
	public double getError(Instances data) throws Exception{
		eval.evaluateModel(classifier, data);
		return eval.pctIncorrect();
	}
	
	private double getError(AbstractClassifier baseClassifier, Instance ins) throws Exception{
		int response = (int)ins.classValue();
		return 1-baseClassifier.distributionForInstance(ins)[response];
	}
}
