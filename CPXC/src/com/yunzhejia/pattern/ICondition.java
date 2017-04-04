/**
 * 
 */
package com.yunzhejia.pattern;

import weka.core.Instance;

/**
 * Define the condition for a specific feature/attribute
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 */
public interface ICondition {
	/**
	 * Test whether this condition holds true for an attribute value of the instance
	 * @param ins the instance for testing 
	 * @return true if thie condition holds true, and false otherwise. 
	 * @throws Throw an exception is the value type does not match.
	 */
	public boolean match(Instance ins);
	
	public int getAttrIndex();
}
