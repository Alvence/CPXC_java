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

public class ADT_Oracle extends AbstractClassifier{
	private static final long serialVersionUID = 3636935337536598456L;
	
	public static final int LE_LABEL = 0;
	public static final int SE_LABEL = 1;
	
	/** type of base classifier*/
	protected ClassifierType baseType = ClassifierType.NAIVE_BAYES; 
	/** type of local classifiers*/
	protected ClassifierType ensembleType = ClassifierType.DECISION_TREE;
	/** type of decision classifier*/
	protected ClassifierType desicionType = ClassifierType.RANDOM_FOREST;
	/** ratio to divide dataset to LargeErrSet and SmallErrSet*/
	protected double rho = 0.5; 
	
	protected int layer;
	
	protected transient AbstractClassifier baseClassifier;
	public transient AbstractClassifier desicionClassifier;
	protected transient AbstractClassifier LEClassifier;
	protected transient AbstractClassifier SEClassifier;
	
	protected Instances oracleLE;
	protected Instances oracleSE;
	protected Instances oracleData;
	
	public void setOracleData(Instances oracleData){
		this.oracleData = oracleData;
	}
	
	public void trainOracle(Instances data) throws Exception{
		oracleLE = new Instances(data,0);
		oracleSE = new Instances(data,0);
		
		AbstractClassifier oracleClassifier = ClassifierGenerator.getClassifier(baseType);
		//step 1 learn a base classifier 
		oracleClassifier.buildClassifier(data);
		
		//step 2 divide D into LE and SE
		List<Double> errs = new ArrayList<Double>();
		double[][] dist = oracleClassifier.distributionsForInstances(data);
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			int label = (int)ins.classValue();
			errs.add(1-dist[i][label]);
		}
		//get cutting point
		double k = cuttingPoint(errs);
		//initialize two data sets
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			if (errs.get(i) >= k){
				oracleLE.add(ins);
			}else{
				oracleSE.add(ins);
			}
		}
	}
	
	public int getOracleLabel(Instance instance) throws Exception{
		if(oracleLE.contains(instance)){
			return LE_LABEL;
		}else if(oracleSE.contains(instance)){
			return SE_LABEL;
		}else{
			throw new Exception("no oracle label found");
		}
	}
	
	//protected transient HashMap<Pattern, LocalClassifier> ensembles;
	
	public ADT_Oracle(int l){
		super();
		layer = l;
	}
	
	public ADT_Oracle(){
		this(0);
	}
	
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances LE = new Instances(data,0);
		Instances SE = new Instances(data,0);
		
		baseClassifier = ClassifierGenerator.getClassifier(baseType);
		desicionClassifier = ClassifierGenerator.getClassifier(desicionType);
		LEClassifier = ClassifierGenerator.getClassifier(ensembleType);
		SEClassifier = ClassifierGenerator.getClassifier(ensembleType);
		//step 1 learn a base classifier 
		baseClassifier.buildClassifier(data);
		
		//step 2 divide D into LE and SE
		divideData(data,LE,SE);
		
		
		
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
		
		
		LEClassifier.buildClassifier(LE);
		SEClassifier.buildClassifier(SE);
		SEClassifier = baseClassifier;
		
//		LEClassifier.buildClassifier(LE);
//		SEClassifier.buildClassifier(SE);
		
		/*if( eval.pctCorrect() < 80 && mergedPLE.numInstances() > 50&& layer < 2){
			System.out.println("aaaa");
			LEClassifier = new ADT(layer+1);
			LEClassifier.buildClassifier(mergedPLE);
		}*/
		
		evaluate(desicionClassifier,newData,"decision");
		testDecisionClassifier(data);
		evaluate(LEClassifier,LE,"LEClassifier");
		evaluate(SEClassifier,SE,"SEClassifier");
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
	private Instances changeHead(Instances data, int total){
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
			newIns.setClassValue(ins.classValue());
			newData.add(newIns);
		}
		
		return newData;
	}
	
	public void testDecisionClassifier(Instances data) throws Exception{
		double err = 0;
		for (Instance ins:data){
			double label = desicionClassifier.classifyInstance(ins);
			int oracle = this.getOracleLabel(ins);
//			System.out.println(label+"  oracle = "+oracle);
			if(Math.abs(label - oracle)>1e-10){
				err+=1;
			}
		}
		err = err/data.numInstances();
		System.out.println("decision testing acc = "+(1-err));
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
	
	
	
	private Instances changeLabel(Instances data, int newLabel){
		Instances newData = new Instances(data,0);
		
		for (Instance ins:data){
			Instance newIns = (Instance)ins.copy();
			newIns.setClassValue(newLabel);
			newData.add(newIns);
		}
		
		return newData;
	}
	
	private Instances extractData(Instances data,AbstractClassifier cl, int label) throws Exception{
		Instances newData = new Instances(data,0);
		for(Instance instance:data){
			if(((int)cl.classifyInstance(instance))==label){
				newData.add(instance);
			}
		}
		return newData;
	}

	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double[] probs = new double[instance.numClasses()];
		for(int i = 0; i < probs.length; i++){
			probs[i] = 0;
		}
		int label = (int)desicionClassifier.classifyInstance(instance);
		
		int oracleLabel = this.getOracleLabel(instance);
//		System.out.println(label+"  oracle = "+oracleLabel);
		if (oracleLabel == SE_LABEL){
			return SEClassifier.distributionForInstance(instance);
		}else if (oracleLabel == LE_LABEL){
			return LEClassifier.distributionForInstance(instance);
		}
		/**/
		double[] probCLS = desicionClassifier.distributionForInstance(instance);
		double[] probLE = LEClassifier.distributionForInstance(instance);
		double[] probSE = SEClassifier.distributionForInstance(instance);
		
		int repL = (int)LEClassifier.classifyInstance(instance);
		int repS = (int)SEClassifier.classifyInstance(instance);
		if (repS == instance.classValue()){
			for(int i = 0; i < probs.length; i++){
				probs[i] = probSE[i];
			}
		}
		else if ( repL == instance.classValue()){
			for(int i = 0; i < probs.length; i++){
				probs[i] = probLE[i];
			}
		}
		else{
			for(int i = 0; i < probs.length; i++){
				probs[i] = probCLS[0] * probLE[i] +  probCLS[1] * probSE[i];
			}
		}		
		
		Utils.normalize(probs);
		return probs;
	}
	
	private double max(double[] arr){
		double max = 0;
		for(double e:arr){
			if(e>max)
				max = e;
		}
		return max;
	}
	
	private void divideData(Instances data, Instances LE, Instances SE) throws Exception{
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
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			if (errs.get(i) >= k){
//			if(Math.random()<0.5){
				LE.add(ins);
			}else{
				SE.add(ins);
			}
		}
		System.out.println("cutting error = " + k);
		
//		Evaluation eval = new Evaluation(data);
//		eval.evaluateModel(baseClassifier, data);
//		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
//		Evaluation eval1 = new Evaluation(data);
//		eval1.evaluateModel(baseClassifier, LE);
//		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%   size="+LE.numInstances());
//		Evaluation eval2 = new Evaluation(data);
//		eval2.evaluateModel(baseClassifier, SE);
//		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%   size="+SE.numInstances());/**/
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
	
	private double cuttingPoint(List<Double> errs) {
//		List<Double> list = new ArrayList<>(errs);
//		Collections.sort(list);
//		return list.get(list.size()/2);
		return 0.2;
	}
	/*
	private double cuttingPoint(List<Double> errs){
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum/errs.size() * rho;
		
		return threshold;
	}
	*/
	public static void main(String[] args){
		ADT_Oracle adt = new ADT_Oracle(5);
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
			
			adt.trainOracle(data);
			adt.setOracleData(data);
			
			Evaluation eval = new Evaluation(data);
			adt.buildClassifier(data);
//			adt.testDecisionClassifier(data);
//			eval.evaluateModel(adt, data);
			eval.crossValidateModel(adt, data, 10, new Random(1));
			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			System.out.println("AUC of "+": " + eval.weightedAreaUnderROC());
			System.out.println(eval.toSummaryString());
			
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.RANDOM_FOREST);
			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
//			eval1.evaluateModel(cl, data);
			eval1.crossValidateModel(cl, data, 10, new Random(1));
			System.out.println("accuracy of NBC: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of NBC: " + eval1.weightedAreaUnderROC());
			/**/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}