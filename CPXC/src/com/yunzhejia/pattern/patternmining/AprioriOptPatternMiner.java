package com.yunzhejia.pattern.patternmining;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.associations.Apriori;
import weka.associations.AprioriItemSet;
import weka.associations.AssociationRule;
import weka.associations.Item;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class AprioriOptPatternMiner implements IPatternMiner {
	Discretizer discretizer = null;
	
	public AprioriOptPatternMiner(Discretizer discretizer){
		this.discretizer = discretizer;
	}
	
	private PatternSet sortBySupport(PatternSet ps, Instances data){
		PatternSet ret = new PatternSet();
		for(IPattern p:ps){
			int index = 0;
			while(ret.size()>index && ret.get(index).support(data)>p.support(data)){
				index++;
			}
			ret.add(index, p);
		}
		return ret;
	}
	
	public IPattern getPattern(Instances data, double minSupp) throws Exception {
		
		
		DataUtils.save(data, "tmp/newData.arff");
		String tmpFile = "tmp/dataForPattern.csv";
		File file = new File(tmpFile);
		
		
		PrintWriter writer = new PrintWriter(file);
		
		for (int i = 0; i < data.numAttributes()-1;i++){
			writer.print(data.attribute(i).name()+ (i==data.numAttributes()-2?"": ","));
		}
		writer.println();
		for(int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			writer.println(discretizer.getDiscretizedInstanceKeepMissingValue(ins,","));
		}
		writer.close();
			
		Instances copy = DataUtils.load(tmpFile);
		copy.setClassIndex(-1);
		
		Filter filter = new NumericToNominal();
		filter.setInputFormat(copy);
		
		copy = Filter.useFilter(copy, filter);
	
//		System.out.println(copy);
		
		PatternSet ps = new PatternSet();
		
		//remove class information
		if(copy.classIndex()!=-1){
			copy.deleteAttributeAt(copy.classIndex());
			copy.setClassIndex(-1);
		}
		
		Apriori apriori = new Apriori();
		apriori.setCar(false);
		apriori.setLowerBoundMinSupport(minSupp);
		apriori.setNumRules(10);
		DataUtils.save(copy, "tmp/newData.arff");
		apriori.buildAssociations(copy);
		
		ArrayList<ArrayList<Object>> itemsets = apriori.getLargeitems();
		
		/*
		AssociationRules rules = apriori.getAssociationRules();
		for(AssociationRule rule:rules.getRules()){
			System.out.println(rule);
			ps.add(parsePattern(rule,data));
		}*/
		for (int i = 0; i < itemsets.size(); i++) {
	          for (int j = 0; j < (itemsets.get(i)).size(); j++) {
	        	  IPattern newPattern = parsePattern((AprioriItemSet) (itemsets.get(i)).get(j)
	        			  ,data,copy);
	        	  ps.add(newPattern);
	          }
	      }
		ps = sortBySupport(ps,data);
		return ps.get(0);
	}
	
	
	@Override
	public PatternSet minePattern(Instances data, double minSupp) throws Exception {
		
		PatternSet ps = new PatternSet();
		Instances tmp = new Instances(data);;

		while(tmp!=null && tmp.size()!=0){
			IPattern p = getPattern(tmp,minSupp);
			Instances d = new Instances(tmp,0);
			for(Instance ins:tmp){
				if (!p.match(ins)){
					d.add(ins);
				}
			}
			ps.add(p);
			tmp = d;
		}
		System.out.println(ps);

		return ps;
	}
	
	private IPattern parsePattern(AprioriItemSet itemSet, Instances headerInfo, Instances discretizedData){
		List<String> items = new ArrayList<>();

		int[] m_items = itemSet.items();
		for (int i = 0; i < discretizedData.numAttributes(); i++) {
		     if (m_items[i] != -1) {
		    	 items.add(discretizedData.attribute(i).value(m_items[i]));
		     }
		}
		
		Set<ICondition> conditions = new HashSet<>();
		for (String item:items){
				int disVal = Integer.parseInt(item);
				int attrIndex = discretizer.getAttributeIndex(disVal);
				ICondition condition = null;
				String attrName = headerInfo.attribute(attrIndex).name();
				if (discretizer.isNumeric(attrIndex)){
					double left = discretizer.getLeft(disVal);
					double right = discretizer.getRight(disVal);
					condition = new NumericCondition(attrName,attrIndex, left, right);
				}else{
					String value = discretizer.getNominal(disVal);
					condition = new NominalCondition(attrName,attrIndex, value);
				}
				conditions.add(condition);
		}
		IPattern pattern = new Pattern(conditions);
		return pattern;
		
	}
	
	private IPattern parsePattern(AssociationRule rule, Instances headerInfo){
		List<String> items = new ArrayList<>();
		for(Item item:rule.getPremise()){
			items.add(item.getItemValueAsString());
		}
		for(Item item:rule.getConsequence()){
			items.add(item.getItemValueAsString());
		}
		
		Set<ICondition> conditions = new HashSet<>();
		for (String item:items){
				int disVal = Integer.parseInt(item);
				int attrIndex = discretizer.getAttributeIndex(disVal);
				ICondition condition = null;
				String attrName = headerInfo.attribute(attrIndex).name();
				if (discretizer.isNumeric(attrIndex)){
					double left = discretizer.getLeft(disVal);
					double right = discretizer.getRight(disVal);
					condition = new NumericCondition(attrName,attrIndex, left, right);
				}else{
					String value = discretizer.getNominal(disVal);
					condition = new NominalCondition(attrName,attrIndex, value);
				}
				conditions.add(condition);
		}
		IPattern pattern = new Pattern(conditions);
		return pattern;
		
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex, boolean flag)
			throws Exception {
		throw new Exception("Unsupport operation");
	}

	public static void main(String[] args) {
		try {
			Instances data = DataUtils.load("data/titanic/train.arff");
			Discretizer discretizer = new Discretizer();
			discretizer.initialize(data);
			
		
			
			AprioriOptPatternMiner miner = new AprioriOptPatternMiner(discretizer);
			PatternSet ps = miner.minePattern(data, 0.1);
			System.out.println("pattern set size="+ps.size());
			for(IPattern p:ps){
				System.out.println(p+" sup= "+p.support(data));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
