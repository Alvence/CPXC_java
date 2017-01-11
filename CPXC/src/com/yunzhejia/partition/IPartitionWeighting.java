package com.yunzhejia.partition;

import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public interface IPartitionWeighting {
	public List<IPartition> calcWeight(List<IPartition> partitions, AbstractClassifier globalCL, Instances validationData) throws Exception ;
}
