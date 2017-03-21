package com.yunzhejia.unimelb.cpexpl.sampler;

import weka.core.Instance;
import weka.core.Instances;

public class SimplePerturbationSampler implements Sampler {

	@Override
	public Instances samplingFromInstance(Instances headerInfo, Instance instance, int N) {
		Instances samples = new Instances(headerInfo, 0);
		
		return null;
	}

	private Instance perturb(Instances headerInfo, Instance instance){
		return null;
	}
}
