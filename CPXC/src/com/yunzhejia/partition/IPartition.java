package com.yunzhejia.partition;

import java.util.List;
import java.util.Set;

import com.yunzhejia.pattern.IPattern;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public interface IPartition {
	public boolean match(Instance ins);
	public boolean isActive();
	public List<Set<IPattern>> getPatternSetList();

	public void setPatternSetList(List<Set<IPattern>> patternSetList);

	public Instances getData();

	public void setData(Instances data);

	public AbstractClassifier getClassifier();

	public void setClassifier(AbstractClassifier classifier);

	public double getWeight() ;

	public void setWeight(double weight);
	public void setActive(boolean active);

}
