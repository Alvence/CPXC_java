package com.yunzhejia.cpxc.pattern;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class AERPatternFilter implements PatternFilter {

	@Override
	public PatternSet filter(PatternSet patternSet) throws Exception {
		List<Double> aers = new ArrayList<>();
		for (Pattern p:patternSet){
			aers.add(p.getLocalClassifier().getWeight());
		}
		double threshold = Collections.max(aers)*0.75;
		PatternSet ps = new PatternSet(patternSet);
		Iterator<Pattern> it = ps.iterator();
		while(it.hasNext()){
			Pattern p = it.next();
			if(p.getLocalClassifier().getWeight() < threshold){
				it.remove();
			}
		}
		return ps;
	}

}
