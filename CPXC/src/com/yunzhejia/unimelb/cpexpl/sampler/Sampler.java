package com.yunzhejia.unimelb.cpexpl.sampler;

import weka.core.Instance;
import weka.core.Instances;

public interface Sampler {
	/**
	 * 
	 * @param headerInfo Header information of the data set
	 * @param instance Instance as the starting point of sampling
	 * @param N number of samples to be generated
	 * @return samples
	 */
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N);
}
