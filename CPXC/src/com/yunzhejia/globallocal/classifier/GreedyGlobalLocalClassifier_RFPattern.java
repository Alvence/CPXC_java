package com.yunzhejia.globallocal.classifier;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.ClustererGenerator.ClustererType;
import com.yunzhejia.partition.IPartition;
import com.yunzhejia.partition.IPartitionWeighting;
import com.yunzhejia.partition.Partition;
import com.yunzhejia.partition.SimulatedAnnealingWeighting;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.GcGrowthPatternMiner;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.ParallelCoordinatesMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class GreedyGlobalLocalClassifier_RFPattern extends AbstractClassifier{
	
	private double minSupp;
	private double minRatio = 3;
	private double rho = 0.9;
	private IPatternMiner patternMiner;
	private transient List<IPartition> partitions;
	private transient AbstractClassifier globalCL;
	
	protected double delta = -1f;
	protected static ClassifierType globalType = ClassifierType.LOGISTIC;
	/** type of decision classifier*/
	protected ClassifierType localType = ClassifierType.LOGISTIC;

	public GreedyGlobalLocalClassifier_RFPattern() {
		this(0.5f,new ParallelCoordinatesMiner(10));
	}
	
	

	public GreedyGlobalLocalClassifier_RFPattern(double minSupp, IPatternMiner patternMiner) {
		this.minSupp = minSupp;
		this.patternMiner = patternMiner;
	}

	private transient Instances trainingData;
	private transient Instances validationData;
	private transient double globalAcc;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		globalCL = ClassifierGenerator.getClassifier(globalType);
		globalCL.buildClassifier(data);
		
		// Make a copy of the data we can reorder
	    Instances datate = new Instances(data);
	    Random random = new Random(1);
	    datate.randomize(random);
	    if (datate.classAttribute().isNominal()) {
	    	datate.stratify(4);
	    }
	    trainingData = datate.trainCV(4, 1, random);
	    validationData = datate.testCV(4, 1);
	    trainingData = datate;
//	    System.out.println("training size="+trainingData.size()+"  validation size="+validationData.size());
//	    System.out.println(validationData);
	    AbstractClassifier tempgcl = ClassifierGenerator.getClassifier(globalType);
	    Evaluation eval = new Evaluation(validationData);
	    tempgcl.buildClassifier(trainingData);
//		adt.testDecisionClassifier(data);
		eval.evaluateModel(tempgcl, validationData);
	    globalAcc = eval.pctCorrect();
	    
	    
		
		patternMiner = new RFPatternMiner();
//		patternMiner = new ManualPatternMiner();
//		patternMiner = new ParallelCoordinatesMiner();
		//2, split it into LE and SE
		Instances LE = new Instances(data,0);
		Instances SE = new Instances(data,0);
		divideData(trainingData,LE,SE);
		Instances newLE = changeLabel(LE, 0);
		Instances newSE = changeLabel(SE, 1);
		Instances decisionData = new Instances(newSE);
		for(Instance ins:newLE){
			decisionData.add(ins);
		}
//		writeData(LE);
		PatternSet ps = patternMiner.minePattern(decisionData, minSupp * trainingData.numInstances());
		System.out.println(ps.size());
//		int c=1;
//		for(IPattern p : ps){
//			System.out.println((c++)+":   "+p);
//		}
//		partitions = pairwisePartition(ps,trainingData);
		partitions = singlewisePartition(ps,trainingData);

		partitions = contrastPartition(partitions, LE, SE);
		
		System.out.println(partitions.size());
//		partitions = bruteForceWeight(partitions);
//		partitions = mergePartition(partitions);
//		System.out.println(partitions.size());
		partitions = filterPartition(partitions);
		if(partitions.size()>0){
			IPartitionWeighting weighter = new SimulatedAnnealingWeighting(1000);
			partitions = weighter.calcWeight(partitions, tempgcl, trainingData);
		}
		System.out.println(partitions.size());
		for (IPartition par:partitions){
			if(par.getWeight()!=0)
			System.out.println(par);
		}
//		
		
//		System.out.println("size="+partitions.size());
//		globalCL = ClassifierGenerator.getClassifier(globalType);
//		globalCL.buildClassifier(trainingData);
		Instances globalData = getGlobalData(trainingData);
		if(globalData.size()>0){
			globalCL.buildClassifier(globalData);
		}else{
			System.out.println("No training Data for global");
			globalCL.buildClassifier(trainingData);
		}
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
	
	private List<IPartition> mergePartition(List<IPartition> partitions) throws Exception {
		List<IPartition> ret = new ArrayList<>();
		int it1Size = partitions.size();
		int it2Size = partitions.size();
		do{
			it1Size = partitions.size();
			partitions = partitionMerge(partitions);
			it2Size = partitions.size();
		}while(it1Size != it2Size);
		return ret;
	}

	private List<IPartition> contrastPartition(List<IPartition> partitions, Instances LE, Instances SE) {
		List<IPartition> ret = new ArrayList<>();
		for(IPartition partition:partitions){
			double ratio = getMDS(partition, LE).size() *1.0 / getMDS(partition,SE).size();
			if(ratio >= minRatio){
				ret.add(partition);
			}
		}
		return ret;
	}

	private void writeData(Instances data){
		 Writer writer;
		try {
			writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream("tmp/LE"), "UTF-8"));
			for (Instance ins:data){
				writer.write(ins.value(0)+","+ins.value(1)+"\n");
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
	    
	}


	private void divideData(Instances data, Instances LE, Instances SE) throws Exception{
		List<Double> errs = new ArrayList<Double>();
		double[][] dist = globalCL.distributionsForInstances(data);
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
			if (errs.get(i) > k){
				LE.add(ins);
			}else{
				SE.add(ins);
			}
		}
		System.out.println("cutting error = " + k);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(globalCL, data);
		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
		Evaluation eval1 = new Evaluation(data);
		eval1.evaluateModel(globalCL, LE);
		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%   size="+LE.numInstances());
		Evaluation eval2 = new Evaluation(data);
		eval2.evaluateModel(globalCL, SE);
		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%   size="+SE.numInstances());/**/
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
	
	private double cuttingPoint(List<Double> errs){
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum/errs.size() * rho;
		
		return threshold;
	}
	private List<IPartition> pairwisePartition(PatternSet ps, Instances data) throws Exception {
		List<IPartition> partitions = new ArrayList<>();
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
				newPartition.setClassifier(ClassifierGenerator.getClassifier(localType));
				newPartition.setData(partitionData);;
				newPartition.setPatternSetList(localPatternSetList);;
				newPartition.getClassifier().buildClassifier(partitionData);
				newPartition.setWeight(newPartition.getData().size()*1.0 / data.size());;
				if(newPartition.getData().size()>=data.numAttributes())
					partitions.add(newPartition);
			}
		}
		return partitions;
	}
	
	private List<IPartition> clusterPartition(ClustererType clustererType, Instances data) throws Exception {
		List<IPartition> partitions = new ArrayList<>();
		
		
		return partitions;
	}

	//find the partition corresponding to a single pattern
	private List<IPartition> singlewisePartition(PatternSet ps, Instances data) throws Exception {
		List<IPartition> partitions = new ArrayList<>();
		for(IPattern pattern:ps){
			Instances partitionData =  getMDS(pattern, data);
			if (partitionData==null || partitionData.size()==0){
				continue;
			}
			Set<IPattern> localPatterns = new HashSet<>();
			localPatterns.add(pattern);
			List<Set<IPattern>> localPatternSetList = new ArrayList<>();
			localPatternSetList.add(localPatterns);
			Partition newPartition = new Partition();
			newPartition.setClassifier(ClassifierGenerator.getClassifier(localType));
			newPartition.setData(partitionData);;
			newPartition.setPatternSetList(localPatternSetList);;
			newPartition.getClassifier().buildClassifier(partitionData);
			newPartition.setWeight(eval(newPartition) - globalAcc);
			if(newPartition.getData().size()>=data.numAttributes() && newPartition.getWeight()>0)
				partitions.add(newPartition);
		}
		return partitions;
	}

	//remove global optimal partitions
	private List<IPartition> filterPartition(List<IPartition> partitions) throws Exception{
		List<IPartition> ret = new ArrayList<>();
		for(IPartition partition:partitions){
			if(!canRemove(partition)){
				ret.add(partition);
			}
		}
		return ret;
	}
	
	private boolean canRemove(IPartition partition) throws Exception{
		double eval = eval(partition);
//		System.out.println("for partition "+partition+"   acc="+eval(partition)+"   while global="+globalAcc);
		if(eval > globalAcc - delta){
//			System.out.println("for partition "+partition+"   acc="+eval(partition)+"   while global="+globalAcc);
			return false;
		}
		
		return true;
	}
	
	private Instances getGlobalData(Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			boolean flag = false;
			for (IPartition par:partitions){
				if(par.getWeight()>0&&par.match(ins)){
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
	
	private Instances getMDS(IPattern patttern,  Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			if(patttern.match(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private Instances getMDS(IPartition partition,  Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			if(partition.match(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	private List<IPartition> partitionMerge(List<IPartition> partitions) throws Exception{
		List<IPartition> ret = new ArrayList<>();
		boolean [] flags = new boolean[partitions.size()];
		for(int i = 0; i < flags.length; i++){
			flags[i] = false;
		}
		
		double maxAdd = 0;
		int a= 0,b= 0;
		
		for (int i = 0; i < partitions.size(); i++){
			for (int j = i + 1; j < partitions.size(); j++){
				if (flags[i]||flags[j]){
					continue;
				}
				IPartition par1 = partitions.get(i);
				IPartition par2 = partitions.get(j);
				
//				if (canMerge(par1,par2)){
//					flags[i] = true;
//					flags[j] = true;
//					Partition newpar = merge(par1,par2);
//					ret.add(newpar);
//				}
				double add = eval(merge(par1,par2)) - eval(par1,par2);
				if(add > maxAdd){
					maxAdd = add;
					a = i;
					b = j;
//					flags[i] = true;
//					flags[j] = true;
//					Partition newpar = merge(par1,par2);
//					ret.add(newpar);
				}
			}
		}
		if(maxAdd > 1){
			flags[a] = true;
			flags[b] = true;
			System.out.println("Combine "+partitions.get(a)+"   and  "+partitions.get(b));
			IPartition newpar = merge(partitions.get(a),partitions.get(b));
			ret.add(newpar);
		}
		for (int i = 0; i < flags.length; i++){
			if (!flags[i]){
				ret.add(partitions.get(i));
			}
		}
		return ret;
	}
	
	private IPartition merge(IPartition par1, IPartition par2) throws Exception{
		IPartition newPartition = new Partition();
		List<Set<IPattern>> ps = new ArrayList<>();
		for(Set<IPattern> p:par1.getPatternSetList()){
			ps.add(p);
		}
		for(Set<IPattern> p:par2.getPatternSetList()){
			ps.add(p);
		}
		Instances data = new Instances(par1.getData(), 0);
		for(Instance ins:par1.getData()){
			data.add(ins);
		}
		for(Instance ins:par2.getData()){
			data.add(ins);
		}
		newPartition.setClassifier(ClassifierGenerator.getClassifier(localType));
		newPartition.setPatternSetList(ps);
		newPartition.setData(data);
		newPartition.setWeight(par1.getWeight()+par2.getWeight());
		newPartition.getClassifier().buildClassifier(data);
		return newPartition;
	}
	
	private boolean canMerge(Partition par1, Partition par2) throws Exception{
		IPartition mergedPar = merge(par1,par2);
		
		double eval1 = eval(par1,par2);
		double evalM = eval(mergedPar);
	
		if(evalM >= eval1 - delta){
			return true;
		}

		return false;
	}
	
	private double eval(IPartition partition) throws Exception{
		Instances globalData = new Instances(trainingData,0);
		for(Instance ins:trainingData){
			if(!partition.match(ins)){
				globalData.add(ins);
			}
		}
		AbstractClassifier gcl = ClassifierGenerator.getClassifier(globalType);
		gcl.buildClassifier(globalData);
		
		
		int correct = 0;
		double acc = 0;
		double[] probs;
		for(Instance testIns:validationData){
			double pre = 0;
			if(partition.match(testIns)){
				pre = partition.getClassifier().classifyInstance(testIns);
				probs = partition.getClassifier().distributionForInstance(testIns);
			}else{
				pre = gcl.classifyInstance(testIns);
				probs = gcl.distributionForInstance(testIns);
			}
			if(pre == testIns.classValue()){
				correct ++;
				acc += maxValue(probs);
			}else{
				acc += probs[(int)testIns.classValue()];
			}
		}
		
		return correct*100.0/validationData.size();
//		return acc/validationData.size();
	}
	
	private double maxValue(double[] chars) {
		double max = chars[0];
	    for (int ktr = 0; ktr < chars.length; ktr++) {
	        if (chars[ktr] > max) {
	            max = chars[ktr];
	        }
	    }
	    return max;
	}
	
	private double eval(IPartition par1, IPartition par2) throws Exception{
		Instances globalData = new Instances(trainingData,0);
		for(Instance ins:trainingData){
			if((!par1.match(ins))&&(!par2.match(ins))){
				globalData.add(ins);
			}
		}
		AbstractClassifier gcl = ClassifierGenerator.getClassifier(globalType);
		gcl.buildClassifier(globalData);
		
		
		int correct = 0;
		for(Instance testIns:validationData){
			double pre = 0;
			if(par1.match(testIns)){
				pre = par1.getClassifier().classifyInstance(testIns);
			} else if(par1.match(testIns)){
				pre = par2.getClassifier().classifyInstance(testIns);
			}else{
				pre = gcl.classifyInstance(testIns);
			}
			if(pre == testIns.classValue()){
				correct ++;
			}
		}
		
		return correct*100.0/validationData.size();
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double[] probs = new double[instance.numClasses()];
		boolean flag = false;
		for(int i = 0; i < probs.length; i++){
			probs[i] = 0;
		}
		for (IPartition par:partitions){
			if(par.getWeight()>0.0&&par.isActive() && par.match(instance)){
				probs = add(probs, par.getClassifier().distributionForInstance(instance), par.getWeight());
				flag = true;
//				return par.classifier.distributionForInstance(instance);
			}
		}
		if (!flag){
			return globalCL.distributionForInstance(instance);
		}
		Utils.normalize(probs);
		return probs;
	}
	
	private double[] add(double[] arr1, double[] arr2, double w){
		if (arr1.length != arr2.length){
			System.err.println("Sizes do not match!!!");
		}
		for (int i = 0; i < arr1.length; i++){
			arr1[i] += arr2[i]*w;
		}
		return arr1;
	}
	
		
	public static void main(String[] args){
		int bestNumBin = -1;
		double bestAcc = 0;
		double bestAUC = 0;
		
		
		try {
			DataSource source;
			Instances data;
	
			source = new DataSource("data/mushroom.arff");
//			source = new DataSource("data/banana.arff");
//			source = new DataSource("data/iris.arff");
			data = source.getDataSet();
		
			
//			for (int bin = 2; bin < 30; bin+=2){
			int bin = 20;
			System.out.println(bin);
			GreedyGlobalLocalClassifier_RFPattern adt = new GreedyGlobalLocalClassifier_RFPattern(0.05f,new ParallelCoordinatesMiner(bin));
			
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
			Evaluation eval = new Evaluation(data);
//			adt.buildClassifier(data);
//			adt.testDecisionClassifier(data);
//			eval.evaluateModel(adt, data);
//			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			eval.crossValidateModel(adt, data, 10, new Random(1));
			
			if (eval.pctCorrect() > bestAcc){
				bestNumBin = bin;
				bestAcc = eval.pctCorrect();
				bestAUC = eval.weightedAreaUnderROC();
				}
//			}
			
//			System.out.println(eval.toSummaryString());
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(GreedyGlobalLocalClassifier_RFPattern.globalType);
//			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
//			eval1.evaluateModel(cl, data);
			eval1.crossValidateModel(cl, data, 10, new Random(1));
			System.out.println("accuracy of "+": " + bestAcc + "%");
			System.out.println("AUC of "+": " + bestAUC);
			System.out.println("accuracy of global: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of global: " + eval1.weightedAreaUnderROC()+"  bin="+bestNumBin);
			/*cl.buildClassifier(data);
			
			    Writer writer = new BufferedWriter(new OutputStreamWriter(
			              new FileOutputStream("tmp/res"), "UTF-8"));
			    Writer writer2 = new BufferedWriter(new OutputStreamWriter(
			              new FileOutputStream("tmp/res2"), "UTF-8"));
			    for (double x = 0; x < 20; x+=0.1){
			    	for(double y = -10; y < 45; y+=0.1){
			    		Instance newIns = data.firstInstance();
			    		newIns.setValue(0, x);
			    		newIns.setValue(1, y);
			    		if (cl.classifyInstance(newIns)==1){
			    			writer.write(x+","+y+"\n");
			    		}
			    		if (cl.classifyInstance(newIns)==0){
			    			writer2.write(x+","+y+"\n");
			    		}
			    	}
			    }
			    writer.close();
			    writer2.close();
		
			
			
			*/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
