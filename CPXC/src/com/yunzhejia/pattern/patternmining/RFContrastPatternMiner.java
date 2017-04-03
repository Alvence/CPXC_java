package com.yunzhejia.pattern.patternmining;

import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instance;
import weka.core.Instances;

public class RFContrastPatternMiner implements IPatternMiner {

	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		return null;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) throws Exception {
		IPatternMiner rfMiner = new RFPatternMiner();
		PatternSet ps = rfMiner.minePattern(data, minSupp);
		
		Instances pos = new Instances(data,0);
		Instances neg = new Instances(data,0);
		for (Instance ins:data){
			if(ins.classValue()==classIndex){
				pos.add(ins);
			}else{
				neg.add(ins);
			}
		}
		PatternSet newPs = new PatternSet();
		for(IPattern p:ps){
			if(p.support(neg)!=0){
				if(p.support(pos)/p.support(neg)>=minRatio){
					p.setRatio(p.support(pos)/p.support(neg));
					newPs.add(p);
				}
			}else{
				if(p.support(pos)!=0){
					p.setRatio(Double.MAX_VALUE);
					newPs.add(p);
				}
			}
		}
		return newPs;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex, boolean flag)
			throws Exception {
		IPatternMiner rfMiner = new RFPatternMiner();
		PatternSet ps = rfMiner.minePattern(data, minSupp);
		
		Instances pos = new Instances(data,0);
		Instances neg = new Instances(data,0);
		for (Instance ins:data){
			if(flag){
				if(ins.classValue()==classIndex){
					pos.add(ins);
				}else{
					neg.add(ins);
				}
			}else{
				if(ins.classValue()!=classIndex){
					pos.add(ins);
				}else{
					neg.add(ins);
				}
			}
		}
		
//		System.out.println("pos="+pos.size()+"   neg="+neg.size() +  "   pps="+ps.size());
		PatternSet newPs = new PatternSet();
		for(IPattern p:ps){
//			System.out.println(p+ "  supp="+p.support(pos) +"  suppneg= "+p.support(neg));
			if(p.support(neg)!=0){
				if(p.support(pos)/p.support(neg)>=minRatio){
					p.setRatio(p.support(pos)/p.support(neg));
					newPs.add(p);
				}
			}else{
				if(p.support(pos)!=0){
					p.setRatio(Double.MAX_VALUE);
					newPs.add(p);
				}
			}
		}
//		System.out.println(newPs.size());
		return newPs;
	}

}
