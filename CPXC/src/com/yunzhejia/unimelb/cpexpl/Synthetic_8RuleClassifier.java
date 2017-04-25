package com.yunzhejia.unimelb.cpexpl;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Synthetic_8RuleClassifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		int numFeature = 8;
			boolean vals[]=new boolean[numFeature];
			for(int a = 0; a< numFeature; a++){
				boolean val = instance.stringValue(a).equals("1");
				vals[a] = val;
			}
			
			boolean result = (vals[0]&&vals[1])||(vals[2]&vals[3]&vals[4])||(vals[5]&vals[6]&vals[7]);
			
			if(result){
				double[] ret = {0,1};
				return ret;
			}else{
				double[] ret = {1,0};
				return ret;
			}
		
	}

}
