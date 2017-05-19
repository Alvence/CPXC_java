package com.yunzhejia.unimelb.cpexpl;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.CPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.FPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.PatternSortingStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.SamplingStrategy;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainerForDNF9 {
	public static void main(String[] args){
//		String[] files = {"balloon.arff","banana.arff", "blood.arff", 
//				"diabetes.arff","haberman.arff","hepatitis.arff","iris.arff","labor.arff",
//				"mushroom.arff","sick.arff","titanic.arff","vote.arff"};
//		String[] files = {"balloon.arff", "blood.arff", "diabetes.arff","haberman.arff","iris.arff","labor.arff"};
//		int[] numsOfExpl = {1,5,10};
//		int[] numsOfSamples={10,200,500,1000};
//		CPStrategy[] miningStrategies = {CPStrategy.APRIORI,CPStrategy.RF};
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.RANDOM,SamplingStrategy.PATTERN_BASED_RANDOM,SamplingStrategy.PATTERN_BASED_PERTURBATION};
//		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC, ClassifierType.DECISION_TREE};
		
		
		
		String[] files = {"synthetic/balloon_synthetic.arff"};
//		String[] files = {"blood.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {1};
		CPStrategy[] miningStrategies = {CPStrategy.APRIORI};
//		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC};
		int[] numsOfSamples={1000};
//		CPExplainer app = new CPExplainer();
		RandomExplainer app = new RandomExplainer();
		try {
			PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
			for(String file:files){
//			Instances data = DataUtils.load("data/synthetic2.arff");
			Instances data = DataUtils.load("data/"+file);
			int numGoldFeature = data.numAttributes();
			Set<Integer> goldFeatures = new HashSet<>();
//			for(int i = 0; i < numGoldFeature-1; i++){
//				goldFeatures.add(i);
//			}
			
//			Instances data = DataUtils.load("tmp/newData.arff");
//			data = AddNoisyFeatureToData.generateNoisyData(data);
			DataUtils.save(data,"tmp/newwData.arff");
			
			//split the data into train and test
			Instances train = DataUtils.load("data/synthetic/DNF9_noisy_train.arff");
			Instances test = DataUtils.load("data/synthetic/DNF9_noisy_test.arff");
			
			
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
						try{

			
			AbstractClassifier cl = new DNF9Classifier();
			cl.buildClassifier(train);
			double precision = 0;
			double recall = 0;
			double probAvg = 0;
			double probMax = 0;
			double probMin = 0;
			int numExpl = 0;
			int count=0;
//			Instance ins = test.get(1);
//			ins.setValue(0, "1");
//			ins.setValue(1, 0.1);
//			ins.setValue(2, 0.1);
//			ins.setValue(1, "YELLOW");
//			ins.setValue(2, "SMALL");
//			ins.setValue(3, "STRETCH");
//			ins.setValue(4, "ADULT");
//			ins.setClassValue(0);
			
//			goldFeatures = InterpretableModels.getGoldenFeature(type, cl, train);
//			System.out.println(goldFeatures);
			
			
			
			for(Instance ins:test){
				
				goldFeatures = getGoldFeature(ins);
				try{
				List<IPattern> expls = app.getExplanations(FPStrategy.APRIORI, samplingStrategy, 
						miningStrategy, PatternSortingStrategy.OBJECTIVE_FUNCTION_LP,
						cl, ins, train, numOfSamples, 0.15, 3, numOfExpl, false);
				if (expls.size()!=0){
					System.out.println(expls);
					precision += ExplEvaluation.evalPrecisionBest(expls, goldFeatures);
					recall += ExplEvaluation.evalRecallBest(expls, goldFeatures);
					probAvg+= ExplEvaluation.evalProbDiffAvg(expls, cl, train, ins);
					probMax+= ExplEvaluation.evalProbDiffMax(expls, cl, train, ins);
					probMin+= ExplEvaluation.evalProbDiffMin(expls, cl, train, ins);
//					System.out.println(expls.size()+"  precision="+precision);
					numExpl+=expls.size();
					count++;
				}else{
//					System.err.println("No explanations!");
				}
				}catch(Exception e){
					throw e;
//					e.printStackTrace();
				}
			}
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cl, test);
			
			String output = "mining="+miningStrategy+" sampling="+samplingStrategy+" numOfSample="+numOfSamples+"   "+file+"  cl="+type+"  NumExpl="+numOfExpl+"  precision = "+(count==0?0:precision/count)+"  recall = "+(count==0?0:recall/count)+"   acc="+eval.correct()*1.0/test.numInstances()
					+" numExpl="+numExpl*1.0/test.size() + " probAvg= "+probAvg/count+" probMax="+probMax/count+" probMin="+probMin/count;
			System.out.println(output);
			writer.println(output);
			writer.flush();
			}catch(Exception e){
//				throw e;
				e.printStackTrace();
//				continue;
			}
						
			
			/*
			Instance ins = test.get(12);
//			ins.setValue(0, 15.634462);
//			ins.setValue(1, 16.646118);
//			ins.setValue(2, 3);
//			ins.setClassValue(1);
			List<IPattern> expls = app.getExplanations(FPStrategy.RF, SamplingStrategy.PATTERN_BASED_RANDOM, 
					CPStrategy.RF, PatternSortingStrategy.PROBDIFF_AND_SUPP,
					cl, ins, data, 2000, 0.01, 10, 5, true);
			precision += ExplEvaluation.eval(expls, goldFeatures);
			System.out.println(expls.size()+"  precision="+precision);
			*/
			
			
					}}}}}}
			writer.close();
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public static Set<Integer> getGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			ret.add(1);
			ret.add(2);
		}else if (instance.stringValue(0).equals("2")){
			ret.add(3);
			ret.add(4);
		}else if (instance.stringValue(0).equals("3")){
			ret.add(5);
			ret.add(6);
		}else if (instance.stringValue(0).equals("4")){
			ret.add(7);
			ret.add(8);
		}
		return ret;
	}
}
