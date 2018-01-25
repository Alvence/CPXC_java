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

public class CPExplainerForSynthetic3 {
	public static void main(String[] args){
//		String[] files = {"balloon.arff","banana.arff", "blood.arff", 
//				"diabetes.arff","haberman.arff","hepatitis.arff","iris.arff","labor.arff",
//				"mushroom.arff","sick.arff","titanic.arff","vote.arff"};
//		String[] files = {"balloon.arff", "blood.arff", "diabetes.arff","haberman.arff","iris.arff","labor.arff"};
//		int[] numsOfExpl = {1,5,10};
//		int[] numsOfSamples={10,200,500,1000};
//		CPStrategy[] miningStrategies = {CPStrategy.APRIORI,CPStrategy.RF};
//		SamplingStrategy[] samplingStrategies = {SamplingStrategy.RANDOM,SamplingStrategy.PATTERN_BASED_RANDOM,SamplingStrategy.PATTERN_BASED_PERTURBATION};
//		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC, ClassifierType.DECISION_TREE};
		
		
		
		String[] files = {"synthetic3.arff"};
//		String[] files = {"blood.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {5};
		CPStrategy[] miningStrategies = {CPStrategy.RF};
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.DECISION_TREE};
		int[] numsOfSamples={50};
		CPExplainer app = new CPExplainer();
//		RandomExplainer app = new RandomExplainer();
		try {
			PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
			for(String file:files){
//			Instances data = DataUtils.load("data/synthetic2.arff");
			Instances data = DataUtils.load("data/"+file);
			int numGoldFeature = data.numAttributes();
			Set<Set<Integer>> goldFeatureSet = new HashSet<>();
//			for(int i = 0; i < numGoldFeature-1; i++){
//				goldFeatures.add(i);
//			}
			
//			Instances data = DataUtils.load("tmp/newData.arff");
//			data = AddNoisyFeatureToData.generateNoisyData(data);
			DataUtils.save(data,"tmp/newwData.arff");
			
			//split the data into train and test
//			Instances train = DataUtils.load("data/synthetic/balloon_noisy_train.arff");
//			Instances test = DataUtils.load("data/synthetic/balloon_noisy_test.arff");
			
//			Instances train = DataUtils.load("data/synthetic/balloon_noisy_train.arff");
//			Instances test = DataUtils.load("data/synthetic/balloon_noisy_test.arff");
//			train.setClassIndex(train.numAttributes()-1);
//			test.setClassIndex(train.numAttributes()-1);
			Instances train  = data;
			Instances test = data;
			
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
			
						try{

			
			AbstractClassifier cl = new Synthetic3Classifier();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			
			cl.buildClassifier(train);
			System.out.println(cl);
			double precision = 0;
			double recall = 0;
			double probAvg = 0;
			double probMax = 0;
			double probMin = 0;
			double f1=0;
			int numExpl = 0;
			int count=0;
//			Instance ins = test.get(9);
//			ins.setValue(0, "1");
//			ins.setValue(1, 0.1);
//			ins.setValue(2, 0.1);
//			ins.setValue(1, "PURPLE");
//			ins.setValue(2, "LARGE");
//			ins.setValue(3, "DIP");
//			ins.setValue(4, "CHILD");
//			ins.setClassValue(0);
			
//			goldFeatures = InterpretableModels.getGoldenFeature(type, cl, train);
//			System.out.println(goldFeatures);
			
			
			
			for(Instance ins:test){
				
				goldFeatureSet = getGoldFeature(ins);
				System.out.println(goldFeatureSet);
				try{
				List<IPattern> expls = app.getExplanations(FPStrategy.APRIORI, samplingStrategy, 
						miningStrategy, PatternSortingStrategy.SUPPORT,
						cl, ins, train, numOfSamples, 0.15, 2, numOfExpl, true);
				if (expls!=null&&expls.size()!=0){
					System.out.println(expls);
					double tmpprecision = 0.0;
					double tmprecall = 0;
					double tmpprobAvg = 0;
					double tmpprobMax = 0;
					double tmpprobMin = 0;
					double tmpf1 = 0.0;
					for(Set<Integer> goldFeatures:goldFeatureSet){
						tmpprecision += ExplEvaluation.evalPrecisionBest(expls, goldFeatures);
						tmprecall += ExplEvaluation.evalRecallBest(expls, goldFeatures);
						tmpprobAvg += ExplEvaluation.evalProbDiffAvg(expls, cl, train, ins);
						tmpprobMax += ExplEvaluation.evalProbDiffMax(expls, cl, train, ins);
						tmpprobMin += ExplEvaluation.evalProbDiffMin(expls, cl, train, ins);
						tmpf1 += ExplEvaluation.evalF1Best(expls, goldFeatures);
					}
					
					precision += tmpprecision/goldFeatureSet.size();
					recall += tmprecall/goldFeatureSet.size();
					probAvg+= tmpprobAvg/goldFeatureSet.size();
					probMax+= tmpprobMax/goldFeatureSet.size();
					probMin+= tmpprobMin/goldFeatureSet.size();
					f1 += tmpf1/goldFeatureSet.size();
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
			
			String output = "mining="+miningStrategy+" sampling="+samplingStrategy+" numOfSample="+numOfSamples+"   "+file+"  cl="+type+"  NumExpl="+numOfExpl+"  precision = "+(count==0?0:precision/count)+"  recall = "+(count==0?0:recall/count)
					+"  f1 = "+(count==0?0:f1/count)+"   acc="+eval.correct()*1.0/test.numInstances()
					+" numExpl="+numExpl*1.0/count + " probAvg= "+probAvg/count+" probMax="+probMax/count+" probMin="+probMin/count
					+"ExplRate="+count*1.0/test.size();
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
	
	public static Set<Set<Integer>> getGoldFeature(Instance instance){
		Set<Set<Integer>> ret = new HashSet<>();
		if ((instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1"))){
			Set<Integer> temp = new HashSet<>();
			temp.add(0);
			temp.add(1);
			ret.add(temp);
		} if ((instance.stringValue(2).equals("1") && instance.stringValue(3).equals("1"))){
			Set<Integer> temp = new HashSet<>();
					temp.add(2);
					temp.add(3);
					ret.add(temp);
		} if ((instance.stringValue(4).equals("1") && instance.stringValue(5).equals("1"))){
			Set<Integer> temp = new HashSet<>();
					temp.add(4);
					temp.add(5);
					ret.add(temp);
		} if ((instance.stringValue(6).equals("1") && instance.stringValue(7).equals("1"))){
			Set<Integer> temp = new HashSet<>();
					temp.add(6);
					temp.add(7);
					ret.add(temp);
		}
		return ret;
	}
}
