package com.yunzhejia.pattern;

import weka.core.Instance;

public class NumericCondition implements ICondition {

	private int attrIndex;
	private double left;
	private double right;
	
	

	public NumericCondition(int attrIndex, double left, double right) {
		super();
		this.attrIndex = attrIndex;
		this.left = left;
		this.right = right;
	}

	/* (non-Javadoc)
	 * @see com.yunzhejia.pattern.ICondition#match(weka.core.Instance)
	 */
	@Override
	public boolean match(Instance ins) {
		double insVal = ins.value(attrIndex);
		if(insVal<=right && insVal > left){
			return true;
		}
		return false;
	}

}
