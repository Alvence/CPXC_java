/**
 * 
 */
package com.yunzhejia.pattern;

import weka.core.Instance;

/**
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 */
public class NominalCondition implements ICondition {

	private int attrIndex;
	private String value;
	/**
	 * 
	 */
	public NominalCondition(int attrIndex, String value) {
		this.attrIndex = attrIndex;
		this.value = value;
	}

	/* (non-Javadoc)
	 * @see com.yunzhejia.pattern.ICondition#match(weka.core.Instance)
	 */
	@Override
	public boolean match(Instance ins) {
		if (ins.stringValue(attrIndex).equals(value)){
			return true;
		}
		return false;
	}

}
