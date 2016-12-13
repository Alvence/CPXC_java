package com.yunzhejia.globallocal.classifier;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ParallelCoordinatesMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class GreedyGlobalLocalClassifier extends AbstractClassifier{
	
	private float minSupp;
	private IPatternMiner patternMiner;
	private transient List<Partition> partitions;
	private transient AbstractClassifier globalCL;
	
	protected static ClassifierType globalType = ClassifierType.DECISION_TREE;
	/** type of decision classifier*/
	protected ClassifierType localType = ClassifierType.DECISION_TREE;

	public GreedyGlobalLocalClassifier() {
		this(0.1f,new ParallelCoordinatesMiner(10));
	}
	
	

	public GreedyGlobalLocalClassifier(float minSupp, IPatternMiner patternMiner) {
		this.minSupp = minSupp;
		this.patternMiner = patternMiner;
	}



	@Override
	public void buildClassifier(Instances data) throws Exception {
		//1, find the patterns;
		PatternSet ps = patternMiner.minePattern(data, minSupp);
		partitions = new ArrayList<>();
		for (int i = 0; i < ps.size();i++){
			
			for (int j = i + 1; j < ps.size(); j++){
				IPattern patterni = ps.get(i);
				IPattern patternj = ps.get(j);
				if(patterni.equals(patternj)){
					continue;
				}
				Instances partitionData = getMutual(patterni,patternj,data);
				if (partitionData==null || partitionData.size()==0){
					continue;
				}
				Set<IPattern> localPatterns = new HashSet<>();
				localPatterns.add(patternj);
				localPatterns.add(patterni);
				List<Set<IPattern>> localPatternSetList = new ArrayList<>();
				localPatternSetList.add(localPatterns);
				Partition newPartition = new Partition();
				newPartition.data = partitionData;
				newPartition.patternSetList = localPatternSetList;
				newPartition.classifier.buildClassifier(partitionData);
				newPartition.weight = newPartition.data.size()*1.0 / data.size();
				if(newPartition.data.size()>=10)
				partitions.add(newPartition);
			}
		}
		
		int it1Size = partitions.size();
		int it2Size = partitions.size();
		do{
			it1Size = partitions.size();
			partitions = partitionMerge(partitions);
			it2Size = partitions.size();
		}while(it1Size != it2Size);
		System.out.println(partitions.size());
		for (Partition par:partitions){
			System.out.println(par);
		}
		
		
		
		globalCL = ClassifierGenerator.getClassifier(globalType);
		Instances globalData = getGlobalData(data);
		globalCL.buildClassifier(globalData);
	}
	private Instances getGlobalData(Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			boolean flag = false;
			for (Partition par:partitions){
				if(par.match(ins)){
					flag = true;
				}
			}
			if (!flag){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private Instances getMutual(IPattern p1, IPattern p2, Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			if(p1.match(ins)&&p2.match(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private List<Partition> partitionMerge(List<Partition> partitions) throws Exception{
		List<Partition> ret = new ArrayList<>();
		boolean [] flags = new boolean[partitions.size()];
		for(int i = 0; i < flags.length; i++){
			flags[i] = false;
		}
		
		for (int i = 0; i < partitions.size(); i++){
			for (int j = i + 1; j < partitions.size(); j++){
				if (flags[i]||flags[j]){
					continue;
				}
				Partition par1 = partitions.get(i);
				Partition par2 = partitions.get(j);
				
				if (canMerge(par1,par2)){
					flags[i] = true;
					flags[j] = true;
					Partition newpar = merge(par1,par2);
					ret.add(newpar);
				}
			}
		}
		
		for (int i = 0; i < flags.length; i++){
			if (!flags[i]){
				ret.add(partitions.get(i));
			}
		}
		return ret;
	}
	
	private Partition merge(Partition par1, Partition par2) throws Exception{
		Partition newPartition = new Partition();
		List<Set<IPattern>> ps = new ArrayList<>();
		for(Set<IPattern> p:par1.patternSetList){
			ps.add(p);
		}
		for(Set<IPattern> p:par2.patternSetList){
			ps.add(p);
		}
		Instances data = new Instances(par1.data, 0);
		for(Instance ins:par1.data){
			data.add(ins);
		}
		for(Instance ins:par2.data){
			data.add(ins);
		}
		newPartition.patternSetList = ps;
		newPartition.data = data;
		newPartition.weight = par1.weight+par2.weight;
		newPartition.classifier.buildClassifier(data);
		return newPartition;
	}
	
	private boolean canMerge(Partition par1, Partition par2) throws Exception{
		Partition mergedPar = merge(par1,par2);
		
		Evaluation eval1 = new Evaluation(par1.data);
		eval1.crossValidateModel(par1.classifier, par1.data, par1.data.size(), new Random(1));
		double acc1=eval1.pctCorrect();
		
		Evaluation eval2 = new Evaluation(par2.data);
		eval2.crossValidateModel(par2.classifier, par2.data, par2.data.size(), new Random(1));
		double acc2=eval2.pctCorrect();
		
		Evaluation evalM = new Evaluation(mergedPar.data);
		evalM.crossValidateModel(mergedPar.classifier, mergedPar.data, mergedPar.data.size(), new Random(1));
		double accM=evalM.pctCorrect();
		
		if(accM >= (acc1+acc2)/2){
			return true;
		}
		return false;
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double[] probs = new double[instance.numClasses()];
		for(int i = 0; i < probs.length; i++){
			probs[i] = 0;
		}
		for (Partition par:partitions){
			if(par.match(instance)){
				return par.classifier.distributionForInstance(instance);
			}
		}
		return globalCL.distributionForInstance(instance);
	}
	
	public class Partition {
		private List<Set<IPattern>> patternSetList;
		private Instances data;
		private AbstractClassifier classifier;
		private double weight;

		public Partition() {
			classifier = ClassifierGenerator.getClassifier(localType);
		}
		
		public boolean match(Instance ins){
			for (Set<IPattern> patternSet:patternSetList){
				if (match(ins, patternSet)){
					return true;
				}
			}
			return false;
		}
		
		
		public boolean match(Instance ins, Set<IPattern> patterns){
			for (IPattern p:patterns){
				if(!p.match(ins)){
					return false;
				}
			}
			return true;
		}
		
		@Override
		public String toString(){
			String ret = "";
			for (Set<IPattern> patternSet:patternSetList){
				ret+="{";
				for (IPattern p:patternSet){
					ret+= p.toString()+" ";
				}
				ret+="}";
			}
			ret+=" data size="+data.size();
			return ret;
		}

	}
	
	public static void main(String[] args){
		int bestNumBin = -1;
		double bestAcc = 0;
		double bestAUC = 0;
		
		
		try {
			DataSource source;
			Instances data;
	
			source = new DataSource("data/synthetic2.arff");
//			source = new DataSource("data/haberman.arff");
//			source = new DataSource("data/iris.arff");
			data = source.getDataSet();
		
			
//			for (int bin = 2; bin < 30; bin+=2){
			int bin = 10;
				System.out.println(bin);
			GreedyGlobalLocalClassifier adt = new GreedyGlobalLocalClassifier(0.01f,new ParallelCoordinatesMiner(bin));
			
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
			Evaluation eval = new Evaluation(data);
			adt.buildClassifier(data);
//			adt.testDecisionClassifier(data);
//			eval.evaluateModel(adt, data);
			eval.crossValidateModel(adt, data, 10, new Random(1));
			
			if (eval.pctCorrect() > bestAcc){
				bestNumBin = bin;
				bestAcc = eval.pctCorrect();
				bestAUC = eval.weightedAreaUnderROC();
				}
//			}
			System.out.println("accuracy of "+": " + bestAcc + "%");
			System.out.println("AUC of "+": " + bestAUC);
//			System.out.println(eval.toSummaryString());
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(GreedyGlobalLocalClassifier.globalType);
//			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
//			eval1.evaluateModel(cl, data);
			eval1.crossValidateModel(cl, data, 10, new Random(1));
			System.out.println("accuracy of global: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of global: " + eval1.weightedAreaUnderROC()+"  bin="+bestNumBin);
			/**/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
