package com.yunzhejia.pattern.patternmining;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instances;

public class ManualPatternMiner implements IPatternMiner {

	public ManualPatternMiner() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		PatternSet ps = new PatternSet();
		ICondition condition = new NumericCondition("x",0, 2, 4);
		IPattern p1 = new Pattern(condition);
		
		condition = new NumericCondition("x",0, 9, 10);
		IPattern p2 = new Pattern(condition);
		
		condition = new NumericCondition("x",0, 14, 16);
		IPattern p3 = new Pattern(condition);
		
		condition = new NumericCondition("x",0, 10, 14);
		IPattern p4 = new Pattern(condition);
		
		ps.add(p1);
		ps.add(p2);
		ps.add(p3);
		ps.add(p4);
		return ps;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex, boolean flag)
			throws Exception {
		throw new Exception("Unsupport operation");
	}

}
