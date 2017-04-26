package com.yunzhejia.unimelb.cpexpl.patternselection;

import com.yunzhejia.pattern.PatternSet;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public interface IPatternSelection {
	public PatternSet select(Instance x, PatternSet ps, AbstractClassifier cl, int K, Instances data) throws Exception;
}
