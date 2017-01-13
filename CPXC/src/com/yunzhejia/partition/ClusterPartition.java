package com.yunzhejia.partition;

import java.util.List;
import java.util.Set;

import com.yunzhejia.pattern.IPattern;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ClusterPartition implements IPartition {
	private boolean active;
	

	public ClusterPartition() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean match(Instance ins) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isActive() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public List<Set<IPattern>> getPatternSetList() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setPatternSetList(List<Set<IPattern>> patternSetList) {
		// TODO Auto-generated method stub

	}

	@Override
	public Instances getData() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setData(Instances data) {
		// TODO Auto-generated method stub

	}

	@Override
	public AbstractClassifier getClassifier() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setClassifier(AbstractClassifier classifier) {
		// TODO Auto-generated method stub

	}

	@Override
	public double getWeight() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setWeight(double weight) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setActive(boolean active) {
		// TODO Auto-generated method stub

	}

}
