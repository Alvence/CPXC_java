package com.yunzhejia.unimelb.cpexpl;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.CPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.FPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.PatternSortingStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.SamplingStrategy;
import com.yunzhejia.unimelb.cpexpl.truth.DTTruth;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainerForJ48 {
	public static void main(String[] args){
//		String[] files = {"balloon.arff","banana.arff", "blood.arff", 
//				"diabetes.arff","haberman.arff","hepatitis.arff","iris.arff","labor.arff",
//				"mushroom.arff","sick.arff","titanic.arff","vote.arff"};
//		String[] files = {"balloon", "blood", "diabetes","ILPD","iris","labor","planning","sick","titanic"	};
//		int[] numsOfExpl = {1,5,10};
//		int[] numsOfSamples={10,200,500,1000};
//		CPStrategy[] miningStrategies = {CPStrategy.APRIORI,CPStrategy.RF};
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
//		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC, ClassifierType.DECISION_TREE};
		
		int[] ratios = {2,3};
		
		String[] files = {"balloon","blood","breast-cancer","diabetes","glass","iris","labor","titanic","vote"};
//		String[] files = {"anneal"};
//		String[] files = {"chess","adult","crx","sonar","ILPD"};
//		String[] files = {"diabetes.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {5};
		CPStrategy[] miningStrategies = {CPStrategy.APRIORI};
//		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.DECISION_TREE};
		int[] numsOfSamples={2000};
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
			
			//split the data into train and test
//			Instances train= DataUtils.load("data/icdm2017/"+file+".arff");
//			Instances test= train;
			Instances data = train;
			
			int numGoldFeature = data.numAttributes();
			Set<Integer> goldFeatures = new HashSet<>();
			
//			for(int ratio:ratios)
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
						try{
			
			CPExplainer app = new CPExplainer();
//			RandomExplainer app = new RandomExplainer();
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
			cl.buildClassifier(train);
			double precision = 0;
			double recall = 0;
			double f1 = 0;
			double probAvg = 0;
			double probMax = 0;
			double probMin = 0;
			int numExpl = 0;
			int count=0;
//			Instance ins = test.get(10);
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
			
			int c = 0;
			for(Instance ins:test){
				
				goldFeatures = DTTruth.getGoldFeature(cl,ins);
				try{
				List<IPattern> expls = app.getExplanations(FPStrategy.APRIORI, samplingStrategy, 
						miningStrategy, PatternSortingStrategy.OBJECTIVE_FUNCTION_LP,
						cl, ins, train, numOfSamples, 0.15, 3, numOfExpl, false);
				if (expls.size()!=0){
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
//					System.err.println("No explanations!"+ c);
				}
				}catch(Exception e){
					throw e;
//					e.printStackTrace();
				}
				c++;
			}
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cl, test);
//			count = test.size();
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
	
	public static Set<Integer> getGoldFeature(AbstractClassifier cl, Instance instance) throws Exception{
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
