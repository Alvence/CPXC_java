package com.yunzhej.unimelb.ppf;

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
import com.yunzhejia.unimelb.cpexpl.CPExplainer.FPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.PatternSortingStrategy;
import com.yunzhejia.unimelb.cpexpl.patternselection.IPatternSelection;
import com.yunzhejia.unimelb.cpexpl.patternselection.ProbDiffPatternSelection;
import com.yunzhejia.unimelb.cpexpl.patternselection.ProbDiffPatternSelectionLP;
import com.yunzhejia.unimelb.cpexpl.sampler.PatternBasedSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.PatternSpacePerturbationSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;
import com.yunzhejia.unimelb.cpexpl.sampler.SimplePerturbationSampler;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;

public class PPF {
	public enum FPStrategy{RF,GCGROWTH,APRIORI,APRIORI_OPT};
	public enum CPStrategy{RF,GCGROWTH,APRIORI};
	public enum SamplingStrategy{RANDOM,PATTERN_BASED_RANDOM,PATTERN_BASED_PERTURBATION};
	public enum PatternSortingStrategy{SUPPORT, PROBDIFF_AND_SUPP, OBJECTIVE_FUNCTION, OBJECTIVE_FUNCTION_LP,NONE};
	public static boolean DEBUG= false;
	PatternSet ps = null;

	Random rand = new Random(0);
	
	public  Instance featureTweaking(AbstractClassifier f, Instance x, Instances data, double eps, double positiveLabel) throws Exception{
		
		if (! (f instanceof RandomForest)){
			return null;
		}
		Instance ret = null;
		double minCost = Double.MAX_VALUE;
		
//		Classifier[] trees = ((RandomForest)f).getClassifiers();
//		for(Classifier cl:trees){
//			if (f.classifyInstance(x)!=positiveLabel && cl.classifyInstance(x)!=positiveLabel){
				List<IPattern> positivePatterns = getExplanations(FPStrategy.APRIORI, SamplingStrategy.RANDOM, 
						CPStrategy.RF, PatternSortingStrategy.SUPPORT,
						f, x, data, 2000, 0.1, 2, 1000, false, positiveLabel);
				
				for (IPattern p:positivePatterns){
					Instance temp = buildInstanceUsingPattern(x,p,eps);
//					System.out.println("x="+x+"  temp="+temp+"  pattern="+p);
					if (f.classifyInstance(temp)==positiveLabel){
//						System.out.println("foound");
						double c = CostFunction.cost(temp,x);
						if(c < minCost){
							ret = temp;
							minCost = c;
						}
					}
				}
			
//			}
//		}	
		return ret;

	}
	private Instance buildInstanceUsingPattern(Instance x, IPattern p, double eps) {
		Instance y = (Instance)x.copy();
		for (ICondition condition: p.getConditions()){
			int attrIndex = condition.getAttrIndex();
			if (condition instanceof NominalCondition){
				String val = ((NominalCondition) condition).getValue();
				y.setValue(attrIndex, val);
			}else{
				double val = y.value(attrIndex);
				
				double left = ((NumericCondition)condition).getLeft();
				double right = ((NumericCondition)condition).getRight();
				
//				if(val>left && val<right)
//					continue;
				if(left!=Double.MIN_VALUE){
					y.setValue(attrIndex, left+eps);
				}else{
					y.setValue(attrIndex, right-eps);
				}
			}
		}
		
		return y;
	}

	public List<IPattern> getExplanations(FPStrategy fpStrategy, SamplingStrategy samplingStrategy, CPStrategy cpStrategy, PatternSortingStrategy patternSortingStrategy,
			AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K, boolean debug, double positiveLabel) throws Exception{
		//debug mode?
		DEBUG = debug;
		
		List<IPattern> ret = new ArrayList<>();
		if(DEBUG)
			System.out.println("instance being tested: " + instance+" classification="+  cl.classifyInstance(instance)+ "class="+instance.classValue()+"  positiveClass="+positiveLabel);
		//step 1, sample the neighbours from the instance
		/*
		Sampler sampler = new SimplePerturbationSampler();
		*/
		Sampler sampler = null;
		IPatternMiner pm = null;
		Discretizer discretizer0 = new Discretizer();
		switch(samplingStrategy){
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
//			System.out.println(ps.size());
			sampler = new PatternSpacePerturbationSampler(ps, 3);
			break;
		default:
			break;
		}
		Instances samples = sampler.samplingFromInstance(headerInfo, instance, N);
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
//			patternMiner = new RFContrastPatternMiner();
			patternMiner = new RFPatternMiner();
			break;
		case GCGROWTH:
			discretizer.initialize(headerInfo);
			patternMiner = new GcGrowthContrastPatternMiner(discretizer);
			break;
		case APRIORI:
			discretizer.initialize(headerInfo);
//			patternMiner = new AprioriContrastPatternMiner(discretizer);
			patternMiner = new AprioriPatternMiner(discretizer);
			break;
		default:
			break;
		}
		
		
		PatternSet patternSet = new PatternSet();
		while(patternSet.isEmpty()){
//			patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)positiveLabel, true);
			patternSet = patternMiner.minePattern(samples, minSupp);
			minRatio/=2;
		}
		if(DEBUG){
			System.out.println("pattern = "+patternSet.size());
		}
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
			if(patternSet.size()>K)
				patternSet = selector.select(instance, patternSet, cl, K, samples, headerInfo);
			break;
		default:
			break;
		}
		if(DEBUG){
			System.out.println("pattern = "+patternSet.size());
		}
		for(int i = 0; i < patternSet.size()&i<K; i++){
			IPattern p = patternSet.get(i);
			ret.add(p);
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
				if(!p.equals(q)&&p.subset(q) && (predictionByRemovingPattern(cl, x, p, header).prob <= predictionByRemovingPattern(cl, x, q, header).prob )){
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
			int num_sample = 10;
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
				if(ret ==null || (pred.classIndex!=label && pred.prob>max)){
					ret = pred;
					if (pred.classIndex!=label){
						max = pred.prob;
					}
				}
			}
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
	class Prediction{
		double classIndex;
		double prob;
		
		@Override
		public String toString(){
			return "ClassIndex = "+classIndex+"  Prob="+prob;
		}
	}
}
