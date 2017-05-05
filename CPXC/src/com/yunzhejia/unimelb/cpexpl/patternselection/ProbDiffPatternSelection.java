package com.yunzhejia.unimelb.cpexpl.patternselection;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.OverlapCalculator;
import com.yunzhejia.pattern.PatternSet;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ProbDiffPatternSelection implements IPatternSelection {

	private Random rand = new Random(0);
	private int maxIt;
	private double initialTemperature = 100;
	private double temperatureDeclineRate = 0.99;
	
	public ProbDiffPatternSelection(int maxIt){
		this.maxIt = maxIt;
	}
	
	public ProbDiffPatternSelection(){
		this(10000);
	}
	
	@Override
	public PatternSet select(Instance x, PatternSet ps, AbstractClassifier cl,int K, Instances samples, Instances headerInfo) throws Exception {
		List<Boolean> current = new ArrayList<>();
		double temperature = initialTemperature;
		// randomly generate initial weights
		for (int i = 0; i < ps.size(); i++) {
			current.add(false);
		}
		current.set(0, true);
		
		double currentVal = eval(ps, cl, current, x, samples, headerInfo);
		List<Boolean> bestWeights = current;
		double bestVal = currentVal;
		int iteration = 0;
		while (iteration < maxIt) {
			List<Boolean> nei = getNeighbour(current, K);
			double neiVal = eval(ps, cl, nei,x, samples, headerInfo);
			if (neiVal > bestVal) {
				bestVal = neiVal;
				bestWeights = nei;
			}
//			System.out.print("get cur:  ");
//			OutputUtils.print(current);
//			System.out.print("get neighbour:  ");
//			OutputUtils.print(nei);
			if (acceptProposal(currentVal, neiVal, temperature)) {
				current = nei;
				currentVal = neiVal;
			}
			
//			System.out.println("accept? " + (acceptProposal(currentVal, neiVal, temperature) ? "true" : "false"));
			iteration++;
			temperature = temperatureDeclineRate * temperature;
		}
		
		PatternSet ret = new PatternSet();
		for (int i = 0; i < ps.size();i++){
			if(bestWeights.get(i)){
				ret.add(ps.get(i));
			}
		}
		return ret;
	}

	private double prediction(AbstractClassifier cl, Instance instance, double classIndex) throws Exception{
		return  cl.distributionForInstance(instance)[(int)classIndex];
	}
	
	private double getRand(double lower, double upper){
		return lower + rand.nextDouble()*(upper-lower);
	}
	
	//Get the prediction without features appearing in the pattern
	public double predictionByRemovingPattern(AbstractClassifier cl, Instance instance, IPattern pattern, Instances headerInfo) throws Exception{
				
		Instance ins = (Instance)instance.copy();
		
		List<List<String>> values = new ArrayList<>();
		for(int i = 0; i < instance.numAttributes();i++){
			values.add(new ArrayList<String>());
		}
		
		int numNumericAttr = 5;
		
		for (ICondition condition:pattern.getConditions()){
			if (condition instanceof NominalCondition){
				String val = ((NominalCondition) condition).getValue();
				Enumeration<Object> enums = headerInfo.attribute(condition.getAttrIndex()).enumerateValues();
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
					double lower = headerInfo.attributeStats(condition.getAttrIndex()).numericStats.min;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
				if(right!=Double.MAX_VALUE){
					double upper = headerInfo.attributeStats(condition.getAttrIndex()).numericStats.max;
					double lower = right;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
			}
		}
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
		
		
//		System.out.println(ins);
		int classIndex = (int)cl.classifyInstance(instance);
		
		return prediction(cl,ins,classIndex);
	}	
	
	private int randomIndex(int left, int right) {
		return (int) Math.round(Math.random() * (right - left) + left);
	}
	
	
	private boolean acceptProposal(double current, double proposal, double temperature) {
		double prob;
		if (temperature == 0) {
			return false;
		}
		prob = Math.exp((proposal - current) / temperature);
		
		return Math.random() < prob;
	}

	private List<Boolean> getNeighbour(List<Boolean> current, int k) {
		boolean flag = true;
		while(flag){
			List<Boolean> nei = new ArrayList<>();
		
			for (Boolean d : current) {
				nei.add(d);
			}
			int moveIndex = randomIndex(0, nei.size() - 1);
			boolean val = nei.get(moveIndex);
			nei.set(moveIndex, !val);
			
			moveIndex = randomIndex(0, nei.size() - 1);
			val = nei.get(moveIndex);
			nei.set(moveIndex, !val);
		
			int count = 0;
			for(Boolean d:nei){
				if(d) count++;
			}
//			System.out.println(count+"<"+k);
			if(count<=k && count > 0){
				return nei;
			}
		}
		return null;
	}

	private double eval(PatternSet ps, AbstractClassifier cl, List<Boolean> nei, Instance instance, Instances samples, Instances headerInfo) throws Exception {
		PatternSet tmp = new PatternSet();
		for (int i = 0; i < ps.size();i++){
			if(nei.get(i)){
				tmp.add(ps.get(i));
			}
		}
		
		double L = 0.0;
		double classIndex = cl.classifyInstance(instance);
		double probOriginal = prediction(cl,instance,classIndex);
		for(IPattern p :tmp){
			double probDiff = predictionByRemovingPattern(cl, instance, p, headerInfo);
//			L += p.support(data)*(probOriginal - probDiff);
			L += (probOriginal - probDiff);
		}
		
		
//		if(tmp.size()>0)
//			L=L/tmp.size();
		
		double omega = 0.0;
		int M = 0;
		for(int i = 0; i < tmp.size();i++){
			for(int j = i +1; j < tmp.size();j++){
				omega+= OverlapCalculator.overlapMDS(tmp.get(i), tmp.get(j), samples);
//				omega+= OverlapCalculator.overlap(tmp.get(i), tmp.get(j),headerInfo);
//				System.out.println("mds over="+OverlapCalculator.overlapMDS(tmp.get(i), tmp.get(j), data));
//				System.out.println("over = "+OverlapCalculator.overlap(tmp.get(i), tmp.get(j), data));
				M++;
			}
		}
		omega = omega/M;
		
//		System.out.println("L=  "+L+"   Omega="+omega+"  tmp="+tmp.size()+" tmp:"+tmp +"  obj="+( L - 0.1*omega));
		return L - 0.1*omega;
	}

}
