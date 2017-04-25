package com.yunzhejia.unimelb.cpexpl;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Synthetic_8Classifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		Random random = new Random(1);
		double[] coef = {1.0,1.5,-2.0,3.0,-0.5,3,-2.5,-2,1};
			double sum=0.0;
			for(int a = 0; a< coef.length-1; a++){
				if(!instance.isMissing(a)){
					double val = instance.value(a);
					sum+=val*coef[a];
				}
			}
			sum+= coef[coef.length-1];
			sum+= random.nextGaussian();
			double prob1 = 1/(1+Math.exp(-sum));
			
				double[] ret = {1-prob1,prob1};
				return ret;

		
	}

}
