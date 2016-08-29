package com.yunzhejia.cpxc.pattern;

import java.util.Iterator;

public class SupportPatternFilter implements PatternFilter {
	private int minSup;
	
	public SupportPatternFilter(int minSup) {
		this.minSup = minSup;
	}

	@Override
	public PatternSet filter(PatternSet patternSet) {
		PatternSet newSet = new PatternSet(patternSet);
		Iterator<Pattern> it = newSet.iterator();
		while(it.hasNext()){
			Pattern p = it.next();
			if(p.getSupport() < minSup){
				it.remove();
			}
		}
		return newSet;
	}

}
