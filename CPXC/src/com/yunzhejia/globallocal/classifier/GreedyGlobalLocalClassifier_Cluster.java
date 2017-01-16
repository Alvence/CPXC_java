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

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.ClustererGenerator;
import com.yunzhejia.cpxc.util.ClustererGenerator.ClustererType;
import com.yunzhejia.partition.ClusterPartition;
import com.yunzhejia.partition.ExhausitiveWeighting;
import com.yunzhejia.partition.IPartition;
import com.yunzhejia.partition.IPartitionWeighting;
import com.yunzhejia.partition.Partition;
import com.yunzhejia.partition.SimulatedAnnealingWeightingBinary;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.clusterers.AbstractClusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class GreedyGlobalLocalClassifier_Cluster extends AbstractClassifier{
	
	private double minSupp;
	private double minRatio = 3;
	private double rho = 0.55;
	private IPatternMiner patternMiner;
	private transient List<IPartition> partitions;
	private transient AbstractClassifier globalCL;
	
	protected double delta = -1f;
	protected static ClassifierType globalType = ClassifierType.DECISION_TREE;
	/** type of decision classifier*/
	protected ClassifierType localType = ClassifierType.DECISION_TREE;
	protected ClustererType clustererType = ClustererType.EM;

	public GreedyGlobalLocalClassifier_Cluster() {
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
//	    trainingData = datate;
//	    validationData = datate;
//	    System.out.println("training size="+trainingData.size()+"  validation size="+validationData.size());
//	    System.out.println(validationData);
	    AbstractClassifier tempgcl = ClassifierGenerator.getClassifier(globalType);
	    Evaluation eval = new Evaluation(validationData);
	    tempgcl.buildClassifier(trainingData);
//		adt.testDecisionClassifier(data);
		eval.evaluateModel(tempgcl, validationData);
	    globalAcc = eval.pctCorrect();
	    
	    
//		partitions = pairwisePartition(ps,trainingData);
		partitions = clusterPartition(clustererType,trainingData);

//		partitions = contrastPartition(partitions, LE, SE);
		
		System.out.println(partitions.size());
//		partitions = bruteForceWeight(partitions);
//		partitions = mergePartition(partitions);
//		System.out.println(partitions.size());
//		partitions = filterPartition(partitions);
		if(partitions.size()>0){
//			IPartitionWeighting weighter = new SimulatedAnnealingWeightingBinary(10000);
			IPartitionWeighting weighter = new ExhausitiveWeighting();
			partitions = weighter.calcWeight(partitions, tempgcl, validationData);
		}
		for (IPartition par:partitions){
			System.out.println(par);
		}
//		
		writeData(data, "tmp/clusteredData"+(num++),clusterer);
//		System.out.println("size="+partitions.size());
//		globalCL = ClassifierGenerator.getClassifier(globalType);
		globalCL.buildClassifier(trainingData);
//		Instances globalData = getGlobalData(trainingData);
//		if(globalData.size()>0){
//			globalCL.buildClassifier(globalData);
//		}else{
//			System.out.println("No training Data for global");
//		}
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


	private void writeData(Instances data, String filename, AbstractClusterer clusterer){
		 Writer writer;
		try {
			
			writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(filename), "UTF-8"));
			for (Instance ins:data){
				if(ins.classIndex()!=-1){
					Instances mdata = new Instances(data,0);
					mdata.add(ins);
					weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
					filter.setAttributeIndices("" + (mdata.classIndex() + 1));
					filter.setInputFormat(mdata);
					Instances dataClusterer = Filter.useFilter(mdata, filter);
					ins = dataClusterer.get(0);
				}
				if (partitions.get(clusterer.clusterInstance(ins)).isActive()){
					writer.write(ins.value(0)+","+ins.value(1)+","+clusterer.clusterInstance(ins)+"\n");
				}else{
					writer.write(ins.value(0)+","+ins.value(1)+", -1\n");
				}
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
	    
	}
	
	

	AbstractClusterer clusterer;
	
	private List<IPartition> clusterPartition(ClustererType clustererType, Instances data) throws Exception {
		List<IPartition> partitions = new ArrayList<>();
		clusterer = ClustererGenerator.getClusterer(clustererType);
		
		weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
		 filter.setAttributeIndices("" + (data.classIndex() + 1));
		 filter.setInputFormat(data);
		 Instances dataClusterer = Filter.useFilter(data, filter);

		 clusterer.buildClusterer(dataClusterer);
		int numOfClusters = clusterer.numberOfClusters();
		for (int i = 0; i < numOfClusters; i++){
			IPartition partition = new ClusterPartition(clusterer, i);
			partition.setActive(true);
			partition.setWeight(1);
			partition.setData(new Instances(data,0));
			partitions.add(partition);
		}
		//assign data to partitions
		for (int i = 0; i < data.size();i++){
			int label = clusterer.clusterInstance(dataClusterer.get(i));
			partitions.get(label).getData().add(data.get(i));
		}
		
		for (IPartition par:partitions){
			if(par.getData().size()<1){
				continue;
			}
			AbstractClassifier classifier = ClassifierGenerator.getClassifier(localType);
			classifier.buildClassifier(par.getData());
			par.setClassifier(classifier);
		}
		
		return partitions;
	}
	static int num=0;


	//remove global optimal partitions
	private List<Partition> filterPartition(List<Partition> partitions) throws Exception{
		List<Partition> ret = new ArrayList<>();
		for(Partition partition:partitions){
			if(!canRemove(partition)){
				ret.add(partition);
			}
		}
		return ret;
	}
	
	private boolean canRemove(Partition partition) throws Exception{
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
	
	
	private Instances getMDS(Partition partition,  Instances data){
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
				return par.getClassifier().distributionForInstance(instance);
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

			Writer writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream("tmp/result"), "UTF-8"));
			DataSource source;
			Instances data;
			String[] files = {"data/synthetic2.arff","data/banana.arff","data/anneal.arff","data/blood.arff","data/diabetes.arff",
					"data/hepatitis.arff","data/ILPD.arff","data/iris.arff","data/labor.arff","data/planning.arff","data/sick.arff"};
			for(String file:files){
//			source = new DataSource("data/synthetic2.arff");
			source = new DataSource(file);
//			source = new DataSource("data/iris.arff");
			data = source.getDataSet();
		
			
			GreedyGlobalLocalClassifier_Cluster adt = new GreedyGlobalLocalClassifier_Cluster();
			
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
			Evaluation eval = new Evaluation(data);
//			adt.buildClassifier(data);
//			adt.testDecisionClassifier(data);
//			eval.evaluateModel(adt, data);
//			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			eval.crossValidateModel(adt, data, 10, new Random(1));
			writer.write(file+"\n");
			writer.write("ACC="+eval.pctCorrect()+"\n");
			writer.write("AUC="+eval.weightedAreaUnderROC()+"\n");
			
//			System.out.println(eval.toSummaryString());
			}
			/*AbstractClassifier cl = ClassifierGenerator.getClassifier(GreedyGlobalLocalClassifier_Cluster.globalType);
//			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
//			eval1.evaluateModel(cl, data);
			eval1.crossValidateModel(cl, data, 10, new Random(1));
			System.out.println("accuracy of "+": " + bestAcc + "%");
			System.out.println("AUC of "+": " + bestAUC);
			System.out.println("accuracy of global: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of global: " + eval1.weightedAreaUnderROC()+"  bin="+bestNumBin);*/
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
			//e.printStackTrace();
		}
	}

}
