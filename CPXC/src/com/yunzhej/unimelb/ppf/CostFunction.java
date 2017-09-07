package com.yunzhej.unimelb.ppf;

import java.util.Random;

import weka.core.Instance;

public class CostFunction {
	static double[] beta;
	
	public static void init(int length){
		beta = new double[length];
		Random rand = new Random(1);
		for (int i = 0; i < beta.length; i++){
			beta[i] = rand.nextDouble()*100;
		}

	}
	public static void init(int length, int seed){
		beta = new double[length];
		Random rand = new Random(seed);
		for (int i = 0; i < beta.length; i++){
			beta[i] = rand.nextDouble()*100;
		}

	}
	public static double cost(Instance x, Instance y){
		double cost = 0;
		for (int i = 0 ; i < beta.length;i++){
			if (x.attribute(i).isNumeric()){
				cost += (x.value(i) - y.value(i))*(x.value(i) - y.value(i))*beta[i];
			}else{
				if(!x.stringValue(i).equals(y.stringValue(i))){
					cost+= beta[i];
				}
			}
		}
		return cost;
	}
}
