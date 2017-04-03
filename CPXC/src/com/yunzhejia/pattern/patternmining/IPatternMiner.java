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
	public PatternSet minePattern(Instances data, double minSupp) throws Exception;
	
	public PatternSet minePattern(Instances data, double minSupp, int featureId) throws Exception;
	
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) throws Exception;
	
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex, boolean flag) throws Exception;
}
