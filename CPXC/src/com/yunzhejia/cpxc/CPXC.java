package com.yunzhejia.cpxc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import com.yunzhejia.cpxc.data.Pattern;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SupervisedFilter;
import weka.filters.supervised.attribute.Discretize;

public class CPXC extends AbstractClassifier{
	/** type of base classifier*/
	ClassifierType baseType;
	/** type of local classifiers*/
	ClassifierType ensembleType;
	/** ratio to divide dataset to LargeErrSet and SmallErrSet*/
	double rho; 
	/** min support for contrast patterns*/
	double minSup;
	/** min growth ratio for contrast patterns*/
	double minRatio;
	
	private AbstractClassifier baseClassifier;
	private HashMap<Pattern, LocalClassifier> ensembles;
	
	private Instances LE;
	private Instances SE;

	
	public CPXC(ClassifierType baseType, ClassifierType ensembleType, double rho, double minSup, double minRatio) {
		super();
		this.baseType = baseType;
		this.ensembleType = ensembleType;
		this.rho = rho;
		this.minSup = minSup;
		this.minRatio = minRatio;
	}


	@Override
	public void buildClassifier(Instances data) throws Exception {
		baseClassifier = ClassifierGenerator.getClassifier(baseType);
		//step 1 learn a base classifier 
		baseClassifier.buildClassifier(data);
		//step 2 divide D into LE and SE
		divideData(data);
		//step 3 perform binning & contrast pattern mining
		binData(data);
	}
	
	private void binData(Instances data) throws Exception{
		Discretize discretizer = new Discretize();
		discretizer.setInputFormat(data);
		Instances discretizedData = Discretize.useFilter(data, discretizer);
		
		for (int i = 0; i < data.numAttributes(); i++){
			System.out.println(data.attribute(i).name());
			System.out.println(discretizer.getBinRangesString(i));
		}
		System.out.println(discretizedData);
		
	}
	
	private void divideData(Instances data) throws Exception{
		List<Double> errs = new ArrayList<Double>();
		double[][] dist = baseClassifier.distributionsForInstances(data);
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			int label = (int)ins.classValue();
			errs.add(1-dist[i][label]);
		}
		//get cutting point
		double k = cuttingPoint(errs);
		
		//initialize two data sets
		LE = new Instances(data,0);
		SE = new Instances(data,0);
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			if (errs.get(i) > k){
				LE.add(ins);
			}else{
				SE.add(ins);
			}
		}
		System.out.println("cutting error = " + k);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(baseClassifier, data);
		System.out.println("accuracy on LE: " + eval.pctCorrect() + "%");
		Evaluation eval1 = new Evaluation(data);
		eval1.evaluateModel(baseClassifier, LE);
		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%");
		Evaluation eval2 = new Evaluation(data);
		eval2.evaluateModel(baseClassifier, SE);
		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%");
	}
/*
	private double cuttingPoint(List<Double> errs){
		
		Collections.sort(errs);
		OutputUtils.print(errs);
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum * rho;
		sum = 0f;
		int index = 0;
		while (sum < threshold){
			sum += errs.get(index);
			index++;
		}
		return errs.get(index-1);
	}
	*/
private double cuttingPoint(List<Double> errs){
		List<Double> list = new ArrayList<Double>(errs);
		Collections.sort(list);
		int index = (int)(list.size() * (1-rho));
		return list.get(index);
	}
}
