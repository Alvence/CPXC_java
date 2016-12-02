/**
 * 
 */
package com.yunzhejia.pattern.patternmining;

import com.yunzhejia.pattern.PatternSet;

import weka.core.Instances;

/**
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 */
public interface IPatternMiner {
	/**
	 * Mine the patterns
	 * @param minSupp
	 * @return
	 */
	public PatternSet minePattern(Instances data, double minSupp);
}
