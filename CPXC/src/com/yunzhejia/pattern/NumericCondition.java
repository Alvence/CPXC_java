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
//		if(insVal<=right && insVal > left){
//			return true;
//		}
		if (left==Double.MIN_VALUE && insVal <= right){
			return true;
		} else if (right==Double.MAX_VALUE && insVal > left){
			return true;
		} else if (left!=Double.MIN_VALUE && right!=Double.MAX_VALUE && insVal<=right && insVal > left){
			return true;
		}
		return false;
	}
 
	@Override
	public String toString(){
		return (left==Double.MIN_VALUE?"":left+"<")+ attrName + 
				(right==Double.MAX_VALUE?"":"<="+right);
//		return left+"<"+attrName+"<"+right;
	}


	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + attrIndex;
		result = prime * result + ((attrName == null) ? 0 : attrName.hashCode());
		long temp;
		temp = Double.doubleToLongBits(left);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(right);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		NumericCondition other = (NumericCondition) obj;
		if (attrIndex != other.attrIndex)
			return false;
		if (attrName == null) {
			if (other.attrName != null)
				return false;
		} else if (!attrName.equals(other.attrName))
			return false;
		if (Math.abs(left-other.left)>1e-10)
			return false;
		if (Math.abs(right-other.right)>1e-10)
			return false;
		return true;
	}
	
	
}
