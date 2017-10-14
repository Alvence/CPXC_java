package com.yunzhejia.unimelb.cpexpl;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.AprioriContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.AprioriPatternMiner;
import com.yunzhejia.pattern.patternmining.GcGrowthContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.GcGrowthPatternMiner;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;
import com.yunzhejia.unimelb.cpexpl.patternselection.IPatternSelection;
import com.yunzhejia.unimelb.cpexpl.patternselection.ProbDiffPatternSelection;
import com.yunzhejia.unimelb.cpexpl.patternselection.ProbDiffPatternSelectionLP;
import com.yunzhejia.unimelb.cpexpl.sampler.GradientBasedSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.PatternBasedSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.PatternSpacePerturbationSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;
import com.yunzhejia.unimelb.cpexpl.sampler.SimplePerturbationSampler;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {
	public enum FPStrategy{RF,GCGROWTH,APRIORI,APRIORI_OPT};
	public enum CPStrategy{RF,GCGROWTH,APRIORI};
	public enum SamplingStrategy{RANDOM,PATTERN_BASED_RANDOM,PATTERN_BASED_PERTURBATION, GRADIENT_BASED_SAMPLING};
	public enum PatternSortingStrategy{SUPPORT, PROBDIFF_AND_SUPP, OBJECTIVE_FUNCTION, OBJECTIVE_FUNCTION_LP,NONE};
	public static boolean DEBUG= false;
	PatternSet ps = null;

	Random rand = new Random(0);
	
	public List<IPattern> getExplanations(FPStrategy fpStrategy, SamplingStrategy samplingStrategy, CPStrategy cpStrategy, PatternSortingStrategy patternSortingStrategy,
			AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K, boolean debug) throws Exception{
		//debug mode?
		DEBUG = debug;
		
		List<IPattern> ret = new ArrayList<>();
		if(DEBUG)
			System.out.println("instance being tested: " + instance+" classification="+  cl.classifyInstance(instance));
		//step 1, sample the neighbours from the instance
		/*
		Sampler sampler = new SimplePerturbationSampler();
		*/
		Sampler sampler = null;
		IPatternMiner pm = null;
		Discretizer discretizer0 = new Discretizer();
		switch(samplingStrategy){
		case GRADIENT_BASED_SAMPLING:
			sampler = new GradientBasedSampler();
			break;
		case RANDOM:
			sampler = new SimplePerturbationSampler();
			break;
		case PATTERN_BASED_RANDOM:
			switch(fpStrategy){
			case RF:
				pm = new RFPatternMiner();
				break;
			case GCGROWTH:
				discretizer0.initialize(headerInfo);
				pm = new GcGrowthPatternMiner(discretizer0);
				break;
			case APRIORI:
				discretizer0.initialize(headerInfo);
				pm = new AprioriPatternMiner(discretizer0);
				break;
			default:
				break;
			}
			if(ps==null){
				ps = pm.minePattern(headerInfo, 0.4);
			}
			sampler = new PatternBasedSampler(ps);
			break;
		case PATTERN_BASED_PERTURBATION:
			switch(fpStrategy){
			case RF:
				pm = new RFPatternMiner();
				break;
			case GCGROWTH:
				discretizer0.initialize(headerInfo);
				pm = new GcGrowthPatternMiner(discretizer0);
				break;
			case APRIORI:
				discretizer0.initialize(headerInfo);
				pm = new AprioriPatternMiner(discretizer0);
				break;
			default:
				break;
			}
			if(ps==null){
				ps = pm.minePattern(headerInfo, 0.4);
			}
			System.out.println(ps.size());
			sampler = new PatternSpacePerturbationSampler(ps, ps.size()/100>3?ps.size()/100:3);
			break;
		default:
			break;
		}
		Instances samples = sampler.samplingFromInstance(cl,headerInfo, instance, N);
//		System.out.println(samples);
		
//		Sampler sampler1 = new SimplePerturbationSampler();
//		Instances samples2 = sampler1.samplingFromInstance(headerInfo, instance, N);
//		OverlapCalculation.calcOverlap(samples, samples2);
//		
		//step 2, label the samples using the classifier cl
		samples = labelSample(samples, cl);
		
		
		
		if(DEBUG){
			System.out.println("sample size = "+samples.size());
//			System.out.println(samples);
		}
//		DataUtils.save(samples, "tmp/11.arff");
		
//		List<Instances> tmp = new ArrayList<>();
//		Instances tmp1= new Instances(headerInfo,0);
//		tmp1.add(instance);
//		tmp.add(tmp1);
//		tmp.add(headerInfo);
//		tmp.add(samples);
//		
//		ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(tmp, 1, 2));;
//		
		
		//step 3, mine the contrast patterns from the newly labelled samples.
		Discretizer discretizer = new Discretizer();
		IPatternMiner patternMiner = null;
		switch(cpStrategy){
		case RF:
			patternMiner = new RFContrastPatternMiner();
			break;
		case GCGROWTH:
			discretizer.initialize(headerInfo);
			patternMiner = new GcGrowthContrastPatternMiner(discretizer);
			break;
		case APRIORI:
			discretizer.initialize(headerInfo);
			patternMiner = new AprioriContrastPatternMiner(discretizer);
			break;
		default:
			break;
		}
		
		int classLabel = (int)cl.classifyInstance(instance);
		
		PatternSet patternSet = new PatternSet();
		while(patternSet.isEmpty()){
			patternSet = patternMiner.minePattern(samples, minSupp, minRatio, classLabel, true);
			minRatio/=2;
			if(minRatio<1){
				return null;
			}
		}
//		if(patternSet.size() == 0){
//			System.out.println(instance + " class="+cl.classifyInstance(instance));
//			System.out.println(samples);
//			throw new Exception("Stop");
//		}else{
//			System.out.println(patternSet.size());
//		}
//		System.out.println(samples);
		patternSet = patternSet.getMatchingPatterns(instance);
		
		PatternSet tmp = new PatternSet();
		for(IPattern p:patternSet){
			if(predictionByRemovingPattern(cl, instance, p, headerInfo).prob  < cl.distributionForInstance(instance)[classLabel]){
				tmp.add(p);
			}
		}
//		System.out.println(patternSet.size());
//		tmp.add(patternSet.get(0));
//		tmp.add(patternSet.get(1));
		if(tmp.size()!=0){
			patternSet = tmp;
		}
//		
//		int ind = 1;
//		for(IPattern p:patternSet){
//			System.out.println((ind++)+": "+p+ "  sup=" + p.support(samples)+" ratio="+p.ratio());
//			System.out.println("Original   "+prediction(cl, instance));
//			System.out.println("Without pattern   "+ new PatternEvaluation().predictionByRemovingPattern(cl, instance, p, headerInfo));
//		}
		
		if(DEBUG){
			System.out.println("size of contrast patterns = "+patternSet.size());
		}
		//step 4, select K patterns and convert them to explanations.
		IPatternSelection selector = null;
		switch(patternSortingStrategy){
		case SUPPORT:
			patternSet=sortBySupport(patternSet);
			break;
		case PROBDIFF_AND_SUPP:
			patternSet = sortByProbDiffAndSupp(cl, instance, patternSet, headerInfo);
			break;
		case OBJECTIVE_FUNCTION:
			selector = new ProbDiffPatternSelection(10000);
			if(patternSet.size()>0)
				patternSet = selector.select(instance, patternSet, cl, K, samples, headerInfo);
			break;
		case OBJECTIVE_FUNCTION_LP:
			selector = new ProbDiffPatternSelectionLP();
			patternSet = filterBySubset(patternSet, cl, instance, headerInfo);
			patternSet=sortBySupport(patternSet);
			if(patternSet.size()>100){
				PatternSet temp = new PatternSet();
				for(int i = 0; i < 100;i++){
					temp.add(patternSet.get(i));
				}
				patternSet = temp;
			}
			
			if(DEBUG){
				System.out.println("size of contrast patterns before LP= "+patternSet.size());
			}
			if(patternSet.size()>K)
				patternSet = selector.select(instance, patternSet, cl, K, samples, headerInfo);
			if(DEBUG){
				System.out.println("size of contrast patterns after LP= "+patternSet.size());
			}
			break;
		default:
			break;
		}
		
//		print_pattern(patternSet,K,"positive");
//		for(int i = 0; i < patternSet.size(); i++){
		for(int i = 0; i < patternSet.size(); i++){
			IPattern p = patternSet.get(i);
			if (DEBUG){
				System.out.println(p + "  sup=" + p.support(samples)+" ratio="+p.ratio());
//				System.out.println("With pattern   "+predictionByPattern(cl, instance, p));
				System.out.println("Original   "+prediction(cl, instance));
				System.out.println("Without pattern   "+predictionByRemovingPattern(cl, instance, p,headerInfo));
				System.out.println();
			}
			ret.add(p);
		}
		
		if (DEBUG){
//		patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance), false);
//		patternSet=sortBySupport(patternSet);
//		print_pattern(patternSet,K,"negative");
		}
		
		return ret;
	}
	
	private PatternSet filterBySubset(PatternSet ps, AbstractClassifier cl, Instance x, Instances header) throws Exception{
		PatternSet tmp = new PatternSet(ps);
		Iterator<IPattern> it = ps.iterator();
		while(it.hasNext()){
			IPattern p = it.next();
			for (IPattern q:tmp){
//				System.out.println(p+"    q="+q+"   subset?"+p.subset(q));
				if(!p.equals(q)&&p.subset(q) && (predictionByRemovingPattern(cl, x, p, header).prob >= predictionByRemovingPattern(cl, x, q, header).prob )){
//					System.out.println(q +"  is subset of  "+p);
					it.remove();
					break;
				}
			}
		}
		
		return ps;
	}
	
	public void print_pattern(PatternSet ps, int K, String name){
		System.out.println(name);
		for(int i = 0; i < K && i < ps.size(); i++){
			IPattern p = ps.get(i);
			System.out.println(p + "  sup=" + p.support()+" ratio="+p.ratio());
		}
	}
	
	//Get the prediction using only features appearing in the pattern
	public Prediction predictionByPattern(AbstractClassifier cl, Instance instance, IPattern pattern) throws Exception{
		Prediction pred = new Prediction();
		
		Instance ins = (Instance)instance.copy();

		Set<Integer> attrs = new HashSet<>();
		for (ICondition condition:pattern.getConditions()){
			attrs.add(condition.getAttrIndex());
		}
		
		for (int i = 0 ; i < ins.numAttributes(); i++){
			if (!attrs.contains(i)){
				ins.setMissing(i);
			}
		}
//		System.out.println(ins);
		pred.classIndex = cl.classifyInstance(ins);
		pred.prob = cl.distributionForInstance(ins)[(int)pred.classIndex];
		
		return pred;
	}
	
	
	public Prediction prediction(AbstractClassifier cl, Instance instance) throws Exception{
		Prediction pred = new Prediction();
		pred.classIndex = cl.classifyInstance(instance);
		pred.prob = cl.distributionForInstance(instance)[(int)pred.classIndex];
		return pred;
	}
	
	//Get the prediction without features appearing in the pattern
		public Prediction predictionByRemovingPattern(AbstractClassifier cl, Instance instance, IPattern pattern, Instances data) throws Exception{
			List<List<String>> values = new ArrayList<>();
			for(int i = 0; i < instance.numAttributes();i++){
				values.add(new ArrayList<String>());
			}
			double label = cl.classifyInstance(instance);
			int numNumericAttr = 5;
			
			for (ICondition condition:pattern.getConditions()){
				if (condition instanceof NominalCondition){
					String val = ((NominalCondition) condition).getValue();
					Enumeration<Object> enums = data.attribute(condition.getAttrIndex()).enumerateValues();
					while(enums.hasMoreElements()){
						String o = (String)enums.nextElement();
						if(!o.equals(val)){
							values.get(condition.getAttrIndex()).add(o);
						}
					}
				}else{
					double left = ((NumericCondition)condition).getLeft();
					double right = ((NumericCondition)condition).getRight();
					if(left!=Double.MIN_VALUE){
						double upper = left;
						double lower = data.attributeStats(condition.getAttrIndex()).numericStats.min;
						for (int i = 0; i < numNumericAttr; i++){
							values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
						}
					}
					if(right!=Double.MAX_VALUE){
						double upper = data.attributeStats(condition.getAttrIndex()).numericStats.max;
						double lower = right;
						for (int i = 0; i < numNumericAttr; i++){
							values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
						}
					}
				}
			}
			int num_sample = 100;
			double max = 0;
			double sum = 0;
			Prediction ret = null;
			for(int index = 0; index < num_sample; index++){
				Instance ins = (Instance)instance.copy();
				for(int i = 0; i < values.size();i++){
					if(values.get(i).size()>0){
						String val = values.get(i).get(rand.nextInt(values.get(i).size()));
						if(ins.attribute(i).isNumeric()){
							ins.setValue(i, Double.parseDouble(val));
						}else{
							ins.setValue(i, val);
						}
					}
				}
				Prediction pred = prediction(cl,ins);
				/*if(ret ==null || (pred.classIndex!=label && pred.prob>max)){
					ret = pred;
					if (pred.classIndex!=label){
						max = pred.prob;
					}
				}*/
				if (pred.classIndex != label){
					sum += (1-pred.prob);
				}else{
					sum+= pred.prob;
				}
			}
			ret = new Prediction();
			ret.classIndex = label;
			ret.prob = sum/num_sample;
			/*
			Instances tmp = new Instances(data,0);
			int[] caps = new int[values.size()];
			int[] curs = new int[values.size()];
			for(int i =0; i < values.size();i++){
				caps[i] = (values.get(i).size());
				curs[i] = 0;
			}
			
			int pos = 0;
			while(true){
				if (pos == values.size()){
					break;
				}
				if(curs[pos] == caps[pos]){
					curs[pos] = 0;
					pos++;
				}
			}*/
			
			
//			System.out.println(ins);
			
			return ret;
		}
		private double getRand(double lower, double upper){
			return lower + rand.nextDouble()*(upper-lower);
		}
	private PatternSet sortByProbDiffAndSupp(AbstractClassifier cl, Instance instance, PatternSet ps, Instances data) throws Exception{
		PatternSet ret = new PatternSet();
//		double[] scores = new double[ps.size()];
		Map<IPattern, Double> scores = new HashMap<>(); 
		for(int i = 0; i < ps.size(); i++){
//			scores.put(ps.get(i), predictionByPattern(cl, instance, ps.get(i)).prob);
			scores.put(ps.get(i), prediction(cl, instance).prob - predictionByRemovingPattern(cl, instance, ps.get(i),data).prob );
		}
		
		for(int i = 0; i < ps.size(); i++){
			IPattern p = ps.get(i);
			int index = 0;
			while(ret.size()>index && (scores.get(ret.get(index))+ret.get(index).support()/2) > (scores.get(p)+p.support()/2)){
				index++;
			}
			ret.add(index, p);
		}
		return ret;
	} 
	
	private PatternSet sortBySupport(PatternSet ps){
		PatternSet ret = new PatternSet();
		for(IPattern p:ps){
			int index = 0;
			while(ret.size()>index && ret.get(index).support()>p.support()){
				index++;
			}
			ret.add(index, p);
		}
		return ret;
	}

	private Instances labelSample(Instances samples, AbstractClassifier cl) throws Exception {
		for (Instance ins:samples){
			ins.setClassValue(cl.classifyInstance(ins));
		}
		
		return samples;
	}
	
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
		
		
		
		String[] files = {"titanic/train.arff"};
//		String[] files = {"blood.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {5};
		CPStrategy[] miningStrategies = {CPStrategy.APRIORI};
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.DECISION_TREE};
		int[] numsOfSamples={1000};
		CPExplainer app = new CPExplainer();
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
			
			Random random = new Random(0);
			//split the data into train and test
//			data.randomize(random);
//			Instances train = data.trainCV(5, 0);
//			Instances test = data.testCV(5, 0);
			Instances train = DataUtils.load("data/titanic/train.arff");
			Instances test = DataUtils.load("data/titanic/test.arff");
			
			
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
			
						try{

			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
//			AbstractClassifier cl = new Synthetic_8RuleClassifier();
//			AbstractClassifier cl = new Synthetic_8Classifier();
//			AbstractClassifier cl = new BalloonClassifier();
//			AbstractClassifier cl = new Synthetic3Classifier();
			cl.buildClassifier(train);
			System.out.println(cl);
			double precision = 0;
			double recall = 0;
			double probAvg = 0;
			double probMax = 0;
			double probMin = 0;
			int numExpl = 0;
			int count=0;
			Instance ins = test.get(1);
//			ins.setValue(0, "1");
//			ins.setValue(1, 0.1);
//			ins.setValue(2, 0.1);
//			ins.setValue(1, "YELLOW");
//			ins.setValue(2, "SMALL");
//			ins.setValue(3, "DIP");
//			ins.setValue(4, "ADULT");
//			ins.setClassValue(1);
			
//			goldFeatures = InterpretableModels.getGoldenFeature(type, cl, train);
//			System.out.println(goldFeatures);
			
			
			
//			for(Instance ins:test){
				try{
				List<IPattern> expls = app.getExplanations(FPStrategy.APRIORI, samplingStrategy, 
						miningStrategy, PatternSortingStrategy.OBJECTIVE_FUNCTION_LP,
						cl, ins, data, numOfSamples, 0.15, 3, numOfExpl, true);
				if (expls.size()!=0){
					precision += ExplEvaluation.evalPrecision(expls, goldFeatures);
					recall += ExplEvaluation.evalRecall(expls, goldFeatures);
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
//			}
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
	
	class Prediction{
		double classIndex;
		double prob;
		
		@Override
		public String toString(){
			return "ClassIndex = "+classIndex+"  Prob="+prob;
		}
	}
}
