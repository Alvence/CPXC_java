/**
 * 
 */
package com.yunzhejia.pattern;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Define the data type for a pattern
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 */
public interface IPattern {
	/**
	 * whether the pattern matches the given instance or not
	 * @param ins the instance for testing
	 * @return true if it matches, and false otherwise
	 */
	public boolean match(Instance ins);
	
	public Instances matchingDataSet(Instances data);
	
	public double support(Instances data);
}
