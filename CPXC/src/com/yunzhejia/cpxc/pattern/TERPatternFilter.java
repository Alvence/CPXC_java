package com.yunzhejia.cpxc.pattern;

import java.util.ArrayList;
import java.util.List;

import com.yunzhejia.cpxc.Discretizer;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class TERPatternFilter implements PatternFilter {
	
//	private int maxNum;
	private Instances data;
	private AbstractClassifier baseClassifier;
	private Discretizer discretizer;
	
	
	
	public TERPatternFilter(Instances data, AbstractClassifier baseClassifier, Discretizer discretizer) {
//		this.maxNum = maxNum;
		this.data = data;
		this.baseClassifier = baseClassifier;
		this.discretizer = discretizer;
	}

	@Override
	public PatternSet filter(PatternSet patternSet) throws Exception {
		List<Pattern> cps = new ArrayList<>(patternSet.getPatterns());
		List<Pattern> ps = new ArrayList<>();
		double obj = 0;
		double currentObj = 0;
		do{
			obj = currentObj;
			Pattern pattern = patternMaximizesObj(new ArrayList<>(ps),new ArrayList<>(cps),data, baseClassifier, discretizer);
			if(pattern == null){
				break;
			}
			cps.remove(pattern);
			ps.add(pattern);
			currentObj = new PatternSet(ps).TER(data, baseClassifier, discretizer);
			//System.out.println("currentObj = "+currentObj+"  pre="+obj);
		}while(currentObj - obj > 0.01);
		return new PatternSet(ps);
	}

	private Pattern patternMaximizesObj(List<Pattern> ps, List<Pattern> cps, Instances data,
			AbstractClassifier baseClassifier, Discretizer discretizer) throws Exception {
		double maxObj = new PatternSet(ps).TER(data, baseClassifier, discretizer);
		Pattern ret = null;
		for(Pattern p:cps){
			List<Pattern> tmp = new ArrayList<>(ps);
			tmp.add(p);
			double obj = new PatternSet(tmp).TER(data, baseClassifier, discretizer);
			if(obj > maxObj){
				ret = p;
				maxObj = obj;
			}
		}
		return ret;
	}

}
