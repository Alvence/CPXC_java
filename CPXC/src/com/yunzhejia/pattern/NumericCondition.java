package com.yunzhejia.pattern;

import weka.core.Instance;

public class NumericCondition implements ICondition {

	private int attrIndex;
	private double left;
	private double right;
	private String attrName;
	
	

	public NumericCondition(int attrIndex, double left, double right) {
		this(null, attrIndex, left, right);
	}

	
	public NumericCondition(String attrName, int attrIndex, double left, double right) {
		this.attrIndex = attrIndex;
		this.left = left;
		this.right = right;
		this.attrName = attrName;
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

	@Override
	public String toString(){
		return (left==Double.MIN_VALUE?"":left+"<")+ attrName + 
				(right==Double.MAX_VALUE?"":"<"+right);
//		return left+"<"+attrName+"<"+right;
	}
}
