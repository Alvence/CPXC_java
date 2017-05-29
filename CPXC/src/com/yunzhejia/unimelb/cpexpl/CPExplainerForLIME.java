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
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainerForLIME {
	public static void main(String[] args){
		
		
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
//		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC, ClassifierType.DECISION_TREE};
		
		
		String[] files = {"titanic"};
//		String[] files = {"diabetes.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {100};
		CPStrategy[] miningStrategies = {CPStrategy.APRIORI};
//		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.DECISION_TREE};
		int[] numsOfSamples={2000};
		CPExplainer app = new CPExplainer();
//		RandomExplainer app = new RandomExplainer();
		try {
			PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
			for(String file:files){
//			Instances data = DataUtils.load("data/synthetic2.arff");
			
//			for(int i = 0; i < numGoldFeature-1; i++){
//				goldFeatures.add(i);
//			}
			
//			Instances data = DataUtils.load("tmp/newData.arff");
//			data = AddNoisyFeatureToData.generateNoisyData(data);
			
			//split the data into train and test
			Instances train = DataUtils.load("data/icdm2017Data/"+file+"_train.arff");
			Instances test = DataUtils.load("data/icdm2017Data/"+file+"_test.arff");
			Instances data = train;
			int numGoldFeature = data.numAttributes();
			Set<Integer> goldFeatures = new HashSet<>();
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
			
						try{

			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			cl.buildClassifier(train);
			double precision = 0;
			double recall = 0;
			double probAvg = 0;
			double probMax = 0;
			double probMin = 0;
			double f1 = 0;
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
			
			List<IPattern> exList = LIMEConverter.getAllNominal("data/icdm2017Data/LIME_resutls/"+file,test);
			
			for(int i =0; i < exList.size();i++){
				Instance ins = test.get(i);
				IPattern ex = exList.get(i);
				
				goldFeatures = getDTGoldFeature(cl,test.get(i));
//				System.out.println(test.get(i));
//				System.out.println(goldFeatures);
				try{
				List<IPattern> expls = new ArrayList<>();
				expls.add(ex);
				if (ex!=null){
//					System.out.println(expls);
					precision += ExplEvaluation.evalPrecisionBest(expls, goldFeatures);
					recall += ExplEvaluation.evalRecallBest(expls, goldFeatures);
					f1 += ExplEvaluation.evalF1Best(expls, goldFeatures);
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
			
			String output = "mining="+miningStrategy+" sampling="+samplingStrategy+" numOfSample="+numOfSamples+"   "+file+"  cl="+type+"  NumExpl="+numOfExpl+"  precision = "+(count==0?0:precision/count)+"  recall = "+(count==0?0:recall/count)
					+"  f1 = "+(count==0?0:f1/count)+"   acc="+eval.correct()*1.0/test.numInstances()
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
	
	public static Set<Integer> getDNF9GoldFeature(Instance instance){
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
	
	public static Set<Integer> getDNF3GGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		 
		if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(3);
			ret.add(4);
			
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(5);
			ret.add(6);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(7);
			ret.add(8);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("1")){
			ret.add(9);
			ret.add(10);
		} else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(11);
			ret.add(12);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(13);
			ret.add(14);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(15);
			ret.add(16);
		}else {
			ret.add(17);
			ret.add(18);
		}
		
		return ret;
	}
	public static Set<Integer> getDNFBalloonGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			ret.add(3);
			ret.add(4);
		}else if (instance.stringValue(0).equals("2")){
			ret.add(1);
			ret.add(2);
		}
		return ret;
	}
	public static Set<Integer> getDNF2GGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		if(instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")){
			ret.add(2);
			ret.add(3);
			ret.add(4);
		}
		else if(instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")){
			ret.add(5);
			ret.add(6);
			ret.add(7);
		}
		return ret;
	}
	
	public static Set<Integer> getDTGoldFeature(AbstractClassifier cl, Instance instance) throws Exception{
		if (!(cl instanceof J48)){
			System.err.println("not J48 tree");
			return null;
		}
		J48 dt = (J48)cl;
		ClassifierTree root = dt.m_root;
		int pred = (int)cl.classifyInstance(instance);
		Set<Integer> expl = new HashSet<>();
		while(!root.m_isLeaf){
//			System.out.println(instance.attribute(((C45Split)root.m_localModel).m_attIndex).name());
			int which = root.m_localModel.whichSubset(instance);
			for(int i = 0; i < root.m_sons.length;i++){
				if(i == which)
					continue;
//				System.out.println("tree "+i+" dis="+Arrays.toString(root.m_sons[i].distributionForInstance(instance, false)));
				if(root.m_sons[i].classifyInstance(instance)!=pred){
					expl.add(((C45Split)root.m_localModel).m_attIndex);
				}
			}
			if(which == -1){
				break;
			}
			root = root.m_sons[which];
		}
		return expl;
	}
}
