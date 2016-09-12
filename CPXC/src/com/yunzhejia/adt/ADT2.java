package com.yunzhejia.adt;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import com.yunzhejia.cpxc.CPXC;
import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.pattern.AERPatternFilter;
import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.pattern.PatternFilter;
import com.yunzhejia.cpxc.pattern.PatternSet;
import com.yunzhejia.cpxc.pattern.SupportPatternFilter;
import com.yunzhejia.cpxc.pattern.TERPatternFilter;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.OutputUtils;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class ADT2 extends AbstractClassifier{
	private static final long serialVersionUID = 3636935337536598456L;
	
	public static final double ERR_MARGIN = 0.1;
	
	/** type of base classifier*/
	protected ClassifierType baseType = ClassifierType.NAIVE_BAYES; 
	/** type of local classifiers*/
	protected ClassifierType ensembleType = ClassifierType.DECISION_TREE;
	/** type of decision classifier*/
	protected ClassifierType desicionType = ClassifierType.DECISION_TREE;
	/** ratio to divide dataset to LargeErrSet and SmallErrSet*/
	protected double rho = 0.5; 
	
	protected int layer;
	
	
	protected int totalLabels; 
	protected transient Instances headerInfoData;
	
	protected transient AbstractClassifier baseClassifier;
	public transient AbstractClassifier desicionClassifier;
	protected transient HashMap<Integer, AbstractClassifier> ensembles;
	//protected transient HashMap<Pattern, LocalClassifier> ensembles;
	
	public ADT2(int l, int numEnsembles){
		super();
		layer = l;
		totalLabels = numEnsembles;
	}
	
	public ADT2(){
		this(0,4);
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		int num = totalLabels;
		List<Instances> B = new ArrayList<Instances>(num);
		for(int i = 0 ; i < num; i++){
			B.add(new  Instances(data,0));
		}
		
		baseClassifier = ClassifierGenerator.getClassifier(baseType);
		desicionClassifier = ClassifierGenerator.getClassifier(desicionType);
		ensembles = new HashMap<>();
		//step 1 learn a base classifier 
		baseClassifier.buildClassifier(data);
		
		//step 2 divide D into LE and SE
		divideData(data,B, num);
		
		List<Instances> newList = new ArrayList<Instances>(num);
		for(int i = 0; i < num;i++){
			newList.add(changeLabel(B.get(i),i, num));
		}
		
		Instances newData = new Instances(newList.get(0),0);
		for(Instances instances:newList){
			newData.addAll(instances);
		}
		desicionClassifier.buildClassifier(newData);
		evaluate(desicionClassifier,newData,"desicion");
		headerInfoData = newData;
		
		//partition original data using the decision classifier
		List<Instances> P = new ArrayList<Instances>();
		for(int i = 0 ; i < num; i++){
			P.add(extractData(data, desicionClassifier, i, num, newData));
//			System.out.println(P.get(i).size());
		}
		
		for (int i = 0; i < num; i++){
			Instances trainingData = merge(P.get(i),exclude(B.get(i),P.get(i)));
//			Instances trainingData = B.get(i);
			AbstractClassifier ensemble = ClassifierGenerator.getClassifier(ensembleType);
			ensemble.buildClassifier(trainingData);
			
			Evaluation eval = new Evaluation(trainingData);
			eval.evaluateModel(ensemble, trainingData);
			if( eval.pctCorrect() < 80 && trainingData.numInstances() > 50&& layer < 4){
				System.out.println("aaaa");
				ensemble = new ADT2(layer+1, num);
				ensemble.buildClassifier(trainingData);
			}
			ensembles.put(i, ensemble);
			evaluate(ensemble,trainingData,"C"+i);
		}
		
		evaluate(this,data,"training");
		
		/*
		
		Instances newLE = changeLabel(LE, 0);
		Instances newSE = changeLabel(SE, 1);
		Instances newData = new Instances(newLE);
		newData.addAll(newSE);
		
		desicionClassifier.buildClassifier(newData);
		
		Instances pLE = extractData(data, desicionClassifier, 0);
		Instances pSE = extractData(data, desicionClassifier, 1);
		
		
		Instances L1 = union(LE,pLE);
		Instances S1 = union(SE,pSE);
		Instances L2 = exclude(LE,L1);
		Instances S2 = exclude(SE,S1);
		System.out.println("L1 = "+L1.size()+" L2="+L2.size());
		System.out.println("S1 = "+S1.size()+" S2="+S2.size());
		
		LEClassifier.buildClassifier(merge(pLE,L2));
		SEClassifier.buildClassifier(merge(pSE,S2));
		
//		LEClassifier.buildClassifier(LE);
//		SEClassifier.buildClassifier(SE);
		
		Instances mergedPLE = merge(pLE,L2);
		
		Evaluation eval = new Evaluation(mergedPLE);
		eval.evaluateModel(LEClassifier, merge(pLE,L2));
		if( eval.pctCorrect() < 80 && mergedPLE.numInstances() > 50&& layer < 2){
			System.out.println("aaaa");
			LEClassifier = new ADT2(layer+1);
			LEClassifier.buildClassifier(mergedPLE);
		}
		
		evaluate(desicionClassifier,newData,"decision");
		evaluate(LEClassifier,merge(pLE,L2),"LEClassifier");
		evaluate(SEClassifier,merge(pSE,S2),"SEClassifier");*/
	}
	
	private void evaluate(AbstractClassifier cl, Instances data, String name) throws Exception{
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(cl, data);
		System.out.println("accuracy of "+name+": " + eval.pctCorrect() + "%");
	}
	
	private Instances merge(Instances d1, Instances d2){
		Instances ret = new Instances (d1);
		for(Instance ins:d2){
			if(!ret.contains(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private Instances union(Instances d1, Instances d2){
		Instances ret = new Instances (d1,0);
		for (Instance ins:d1){
			if(d2.contains(ins)){
				ret.add(ins);
			}
		}
		for (Instance ins:d2){
			if(d1.contains(ins) && (!ret.contains(ins))){
				ret.add(ins);
			}
		}
		return ret;
	}
	private Instances exclude(Instances d1, Instances d2){
		Instances ret = new Instances (d1,0);
		for (Instance ins:d1){
			if(!d2.contains(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private Instances changeLabel(Instances data, int newLabel, int total){
		Instances newData = new Instances(data,0);
		List<String> newLabels = new ArrayList<>();
		for(int i = 0; i < total; i++){
			newLabels.add(i+"");
		}
		Attribute newClassAttr = new Attribute("Partition", newLabels);
		
		int classIndex = newData.classIndex();
		
		newData.setClass(newClassAttr);
		newData.deleteAttributeAt(classIndex);
		newData.insertAttributeAt(newClassAttr, classIndex);
		newData.setClassIndex(classIndex);
		
		
		for (Instance ins:data){
			Instance newIns = (Instance)ins.copy();
			newIns.setClassValue(newLabel);
			newData.add(newIns);
		}
		
		return newData;
	}
	
	private Instances extractData(Instances data,AbstractClassifier cl, int label, int numClasses, Instances headerInfoData) throws Exception{
		Instances newData = new Instances(data,0);
		for(Instance instance:data){
			Instance cIns = convertInstance(instance);
			if(((int)cl.classifyInstance(cIns))==label){
				newData.add(instance);
			}
		}
		return newData;
	}
	
	public Instance convertInstance(Instance instance){
		Instance newIns = (Instance) headerInfoData.get(0).copy();
		
		for (int i = 0; i < newIns.numAttributes(); i++){
			if(i != instance.classIndex() && !instance.isMissing(i)){
				if (newIns.attribute(i).isNumeric()){
					newIns.setValue(i, instance.value(i));
				}else{
					newIns.setValue(i, instance.stringValue(i));
				}
			}
		}
		return newIns;
		
	}

	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double[] probs = new double[instance.numClasses()];
		for(int i = 0; i < probs.length; i++){
			probs[i] = 0;
		}
		int label = (int)desicionClassifier.classifyInstance(convertInstance(instance));
		if(true)
		return ensembles.get(label).distributionForInstance(instance);
		
	
		double[] probCLS = desicionClassifier.distributionForInstance(convertInstance(instance));
		OutputUtils.print(probCLS);
		
		for (int j = 0; j < probCLS.length; j++){
			double[] probEnsemble =  ensembles.get(j).distributionForInstance(instance);
			for(int i = 0; i < probs.length; i++){
				probs[i] += probCLS[j] * probEnsemble[i];
			}
		}
		OutputUtils.print(probs);
		Utils.normalize(probs);
		return probs;
	}
	
	
	
	private void divideData(Instances data, List<Instances> instancesList, int num) throws Exception{
		List<Double> errs = new ArrayList<Double>();
		double[][] dist = baseClassifier.distributionsForInstances(data);
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			int label = (int)ins.classValue();
			errs.add(1-dist[i][label]);
		}
		//get cutting points
		double[] points = cuttingPoints(errs, num);
		//initialize two data sets
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			int index = getBin(points,errs.get(i));
//			System.out.println(index);
			instancesList.get(index).add(ins);
//			index = getBinUp(points,errs.get(i));
//			if(index != -1){
//				instancesList.get(index).add(ins);
//			}
//			index = getBinDown(points,errs.get(i));
//			if(index != -1){
//				instancesList.get(index).add(ins);
//			}
		}
		System.out.print("cutting errors = ");
		OutputUtils.print(points);
		/*
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(baseClassifier, data);
		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
		int i = 0;
		for(Instances inss:instancesList){
			Evaluation eval1 = new Evaluation(data);
			eval1.evaluateModel(baseClassifier, inss);
			System.out.println("accuracy on P"+i+++": " + eval1.pctCorrect() + "%" + " size="+inss.size());
		}*/
	}
	
	private int getBin(double[] points, double err){
		int index = 0;
		while(index < points.length && err >= points[index]){
			index++;
		}
		return index;
	}
	
	private int getBinUp(double[] points, double err){
		int index = 0;
		while(index < points.length && err >= points[index]){
			index++;
		}
		if (index == points.length|| err + ERR_MARGIN < points[index]){
			return -1;
		}
		
		return index + 1;
	}
	
	private int getBinDown(double[] points, double err){
		int index = 0;
		while(index < points.length && err >= points[index]){
			index++;
		}
		if (index == 0|| err - ERR_MARGIN > points[index - 1]){
			return -1;
		}
		
		return index - 1;
	}
	
	/*
	private double cuttingPoint(List<Double> errs){
		List<Double> list = new ArrayList<Double>(errs);
		Collections.sort(list);
		//OutputUtils.print(errs);
		double sum = 0f;
		for (double err:list){
			sum += err;
		}
		double threshold = sum * rho;
		
		double calc = 0f;
		int index = list.size()-1;
		while (calc < threshold){
			calc += list.get(index);
			index--;
		}
		return list.get(index);
	}
	*/
	
	private double[] cuttingPoints(List<Double> errs, int num) {
		List<Double> list = new ArrayList<>(errs);
		double[] points = new double[num-1];
		Collections.sort(list);
		int index = 0;
		int stepsize = list.size()/num;
		for(int i = 0; i < num-1;  i++){
			index += stepsize;
			if(index >= list.size()){
				index = list.size() - 1;
			}
			points[i] = list.get(index);
		}
		return points;
	}

	private double[] cuttingPoint(List<Double> errs, int num){
		double[] points = new double[num-1];
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum/errs.size() * rho;
		
		return points;
	}
	
	public static void main(String[] args){
		ADT2 adt = new ADT2(1,2);
		DataSource source;
		Instances data;
		try {
			source = new DataSource("data/ILPD.arff");
//			source = new DataSource("data/vote.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
//			weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
//			discretizer.setInputFormat(data);
//			data = weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
//			System.out.println(data);
			
			Evaluation eval = new Evaluation(data);
//			adt.buildClassifier(data);
//			eval.evaluateModel(adt, data);
			eval.crossValidateModel(adt, data, 7, new Random(1));
			
			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			System.out.println("AUC of "+": " + eval.weightedAreaUnderROC());
			System.out.println(eval.toSummaryString());
			
			/*
			AbstractClassifier cl = new NaiveBayes();
			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
			eval1.evaluateModel(cl, data);
//			eval1.crossValidateModel(cl, data, 7, new Random(1));
			System.out.println("accuracy of NBC: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of NBC: " + eval1.weightedAreaUnderROC());
			*/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}