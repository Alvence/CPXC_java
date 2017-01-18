package com.yunzhejia.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.yunzhejia.cpxc.util.ArrayUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class SimulatedAnnealingWeighting implements IPartitionWeighting {

	private Map<IPartition, Map<Instance, List<Double>>> probsOfParitions;
	private Map<Instance, List<Double>> probsOfGlobal;

	private int maxIt;
	private double stepSize = 0.1;
	private double initialTemperature = 100;
	private double temperatureDeclineRate = 0.99;

	public SimulatedAnnealingWeighting() {
		this(10000);
	}

	public SimulatedAnnealingWeighting(int maxIt) {
		this.maxIt = maxIt;
	}

	@Override
	public List<IPartition> calcWeight(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData)
			throws Exception {

		calcProbs(partitions, globalCL, validationData);
		List<Double> current = new ArrayList<>();
		double temperature = initialTemperature;
		// randomly generate initial weights
		for (int i = 0; i < partitions.size(); i++) {
//			current.add(Math.random()>0.5?1.0:0.0);
			current.add(Math.random());
		}
		System.out.println(current);
		ArrayUtils.normalize(current);
		double currentVal = eval(partitions, globalCL, validationData, current);
		List<Double> bestWeights = current;
		double bestVal = currentVal;
		int iteration = 0;
		while (iteration < maxIt) {
			List<Double> nei = getNeighbour(current);
			double neiVal = eval(partitions, globalCL, validationData, nei);
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

		for (int i = 0; i < partitions.size(); i++) {
			partitions.get(i).setWeight(bestWeights.get(i));
			partitions.get(i).setActive(true);
		}

		return partitions;
	}

	private List<Double> getNeighbour(List<Double> weights) {
		List<Double> nei = new ArrayList<>();
		for (Double d : weights) {
			nei.add(d);
		}
		int moveIndex = randomIndex(0, nei.size() - 1);
		double val = nei.get(moveIndex);

		if (nei.get(moveIndex) == 0) {
			nei.set(moveIndex, val + stepSize);
//			nei.set(moveIndex, 1.0);
		} else {
//			nei.set(moveIndex, 0.0);
			int sign = Math.random() > 0.5 ? -1 : 1;
			double newVal = val + sign * stepSize;
			if (newVal < 0) {
				newVal = 0;
			}
			nei.set(moveIndex, newVal);
		}
		ArrayUtils.normalize(nei);
		return nei;
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
//		System.out
//				.println("proposal=" + proposal + "  current=" + current + "  temp=" + temperature + "   prob=" + prob);
		return Math.random() < prob;
	}

	private void calcProbs(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData)
			throws Exception {
		probsOfGlobal = new HashMap<>();
		probsOfParitions = new HashMap<>();

		for (IPartition partition : partitions) {
			probsOfParitions.put(partition, new HashMap<Instance, List<Double>>());
		}

		for (Instance instance : validationData) {
			// calc probs of globalCL
			List<Double> probOfGlobal = new ArrayList<>();
			probOfGlobal = ArrayUtils.arrayToList(globalCL.distributionForInstance(instance));
			probsOfGlobal.put(instance, probOfGlobal);
		}

		// calc probs of partitions
		for (IPartition par : partitions) {
			Map<Instance, List<Double>> probsOfPartition = new HashMap<>();
			for (Instance instance : validationData) {
				List<Double> probOfPartition = new ArrayList<>();
				probOfPartition = ArrayUtils.arrayToList(par.getClassifier().distributionForInstance(instance));
				probsOfPartition.put(instance, probOfPartition);
			}
			probsOfParitions.put(par, probsOfPartition);
		}
	}

	private double eval(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData,
			List<Double> weights) throws Exception {
		int acc = 0;
		
		Instances globalData = new Instances(validationData,0);
		for(int i = 0; i < partitions.size(); i++){
			if (weights.get(i) != 0.0){
				for(Instance ins:partitions.get(i).getData()){
					globalData.add(ins);
				}
			}
		}
		if (globalData.size()>0){
			AbstractClassifier newGL = (AbstractClassifier)AbstractClassifier.makeCopy(globalCL);
			newGL.buildClassifier(globalData);
			globalCL = newGL;
		}
		
		for (Instance instance : validationData) {
			Double[] probs = new Double[instance.numClasses()];
			boolean flag = false;
			for (int i = 0; i < probs.length; i++) {
				probs[i] = 0.0;
			}
			for (int i = 0; i < partitions.size(); i++) {
				IPartition par = partitions.get(i);
				
				if ( weights.get(i)>0.0&&par.match(instance)) {
					probs = add(probs, probsOfParitions.get(par).get(instance).toArray(probs), weights.get(i));
					// probs = add(probs,
					// par.getClassifier().distributionForInstance(instance),
					// weights.get(i));
					flag = true;
//					System.out.println("ins="+instance+"   par="+par+"  class="+instance.classValue());
//					System.out.println("PAR= "+probsOfParitions.get(par).get(instance));
//					System.out.println("GLO= "+probsOfGlobal.get(instance));
					// return par.classifier.distributionForInstance(instance);
				}
			}
			if (!flag) {
				probs = ArrayUtils.primariesToObjects(globalCL.distributionForInstance(instance));
//				probs = probsOfGlobal.get(instance).toArray(probs);
			}
			double max = 0;
			int c = 0;
			for (int i = 0; i < probs.length; i++) {
				if (probs[i] > max) {
					max = probs[i];
					c = i;
				}
			}
			if (c == instance.classValue()) {
				acc += 1;
			}
		}
//		System.out.println(weights+"   "+acc*100.0/validationData.size());
		return acc * 100.0 / validationData.size();
	}

	private Double[] add(Double[] arr1, double[] arr2, double w) {
		if (arr1.length != arr2.length) {
			System.err.println("Sizes do not match!!!");
		}
		for (int i = 0; i < arr1.length; i++) {
			arr1[i] += arr2[i] * w;
		}
		return arr1;
	}

	private Double[] add(Double[] arr1, Double[] arr2, double w) {
		if (arr1.length != arr2.length) {
			System.err.println("Sizes do not match!!!");
		}
		for (int i = 0; i < arr1.length; i++) {
			arr1[i] += arr2[i] * w;
		}
		return arr1;
	}
}
