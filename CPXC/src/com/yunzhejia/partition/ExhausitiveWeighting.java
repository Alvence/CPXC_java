package com.yunzhejia.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.yunzhejia.cpxc.util.ArrayUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ExhausitiveWeighting implements IPartitionWeighting {

	private Map<IPartition, Map<Instance, List<Double>>> probsOfParitions;
	private Map<Instance, List<Double>> probsOfGlobal;
	
	public ExhausitiveWeighting() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public List<IPartition> calcWeight(List<IPartition> partitions, AbstractClassifier globalCL,
			Instances validationData) throws Exception {
		calcProbs(partitions, globalCL, validationData);
		
		List<IPartition> ret = new ArrayList<>();
		int maxIt = (int)Math.pow(2, partitions.size());
		int size = partitions.size();
		List<Boolean> bestWeights = null;
		double bestAcc = 0;
		for (int i = 1; i <= maxIt; i++){
			List<Boolean> weights = next(i, partitions.size());
			if(weights == null){
				continue;
			}
			for(int j = 0; j < size; j++){
				partitions.get(j).setActive(weights.get(j));;
			}
			double acc = eval(partitions, globalCL, validationData);
			if (bestAcc < acc){
				bestWeights = weights;
				bestAcc = acc;
			}
		}
		for(int j = 0; j < size; j++){
			if (bestWeights.get(j)){
				partitions.get(j).setActive(true);;
			}else{
				partitions.get(j).setActive(false);;
			}
			ret.add(partitions.get(j));
		}
		return ret;
	}
	
	private List<Boolean> next(int i, int size) {
		int nonezero = 0;
		List<Boolean> ret = new ArrayList<>();
		for(int index = 0; index < size; index++){
			int v = i & 1;
			i = i>>1;
			ret.add(v==1);
			if(v==1){
				nonezero += 1;
			}
		}
		if(nonezero > size * .4){
			return null;
		}
		return ret;
	}
	

	private void calcProbs(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData) throws Exception{
		probsOfGlobal = new HashMap<>();
		probsOfParitions = new HashMap<>();
		
		for (IPartition partition: partitions){
			probsOfParitions.put(partition, new HashMap<Instance, List<Double>>());
		}
		
		for (Instance instance: validationData){
			//calc probs of globalCL
			List<Double> probOfGlobal = new ArrayList<>();
			probOfGlobal = ArrayUtils.arrayToList(globalCL.distributionForInstance(instance));
			probsOfGlobal.put(instance, probOfGlobal);
		}
		
		//calc probs of partitions
		for (IPartition par: partitions){
			Map<Instance, List<Double>> probsOfPartition = new HashMap<>();
			for (Instance instance: validationData){
				List<Double> probOfPartition = new ArrayList<>();
				if (par.getClassifier()!=null){
					probOfPartition = ArrayUtils.arrayToList(par.getClassifier().distributionForInstance(instance));
				}else{
					for(int i = 0; i < instance.numClasses();i++){
						probOfPartition.add(0.0);
					}
				}
				probsOfPartition.put(instance, probOfPartition);
			}
			probsOfParitions.put(par, probsOfPartition);
		}
	}
	
	

	private double eval(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData) throws Exception{
		int acc = 0;
		for (Instance instance:validationData){
			Double[] probs = new Double[instance.numClasses()];
			boolean flag = false;
			for(int i = 0; i < probs.length; i++){
				probs[i] = 0.0;
			}
			for (IPartition par:partitions){
				if(par.isActive() && par.match(instance)){
					probs = add(probs, probsOfParitions.get(par).get(instance).toArray(probs), par.getWeight());
					flag = true;
//					return par.classifier.distributionForInstance(instance);
				}
			}
			if (!flag){
				probs = probsOfGlobal.get(instance).toArray(probs);
			}
			double max = 0;
			int c = 0;
			for(int i = 0; i < probs.length; i++){
				if (probs[i]>max){
					max = probs[i];
					c = i;
				}
			}
			if (c == instance.classValue()){
				acc+= 1;
			}
		}
		return acc*100.0/validationData.size();
	}
	private Double[] add(Double[] arr1, Double[] arr2, double w){
		if (arr1.length != arr2.length){
			System.err.println("Sizes do not match!!!");
		}
		for (int i = 0; i < arr1.length; i++){
			arr1[i] += arr2[i]*w;
		}
		return arr1;
	}
	
	/*
	private double eval(List<Partition> partitions, AbstractClassifier globalCL, Instances validationData) throws Exception{
		int acc = 0;
		for (Instance instance:validationData){
			double[] probs = new double[instance.numClasses()];
			boolean flag = false;
			for(int i = 0; i < probs.length; i++){
				probs[i] = 0;
			}
			for (Partition par:partitions){
				if(par.isActive() && par.match(instance)){
					probs = add(probs, par.getClassifier().distributionForInstance(instance), par.getWeight());
					flag = true;
//					return par.classifier.distributionForInstance(instance);
				}
			}
			if (!flag){
				probs = globalCL.distributionForInstance(instance);
			}
			double max = 0;
			int c = 0;
			for(int i = 0; i < probs.length; i++){
				if (probs[i]>max){
					max = probs[i];
					c = i;
				}
			}
			if (c == instance.classValue()){
				acc+= 1;
			}
		}
		return acc*100.0/validationData.size();
	}
	
	private double[] add(double[] arr1, double[] arr2, double w){
		if (arr1.length != arr2.length){
			System.err.println("Sizes do not match!!!");
		}
		for (int i = 0; i < arr1.length; i++){
			arr1[i] += arr2[i]*w;
		}
		return arr1;
	}*/

}
