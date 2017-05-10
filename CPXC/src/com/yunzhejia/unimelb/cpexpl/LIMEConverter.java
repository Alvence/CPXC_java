package com.yunzhejia.unimelb.cpexpl;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.Pattern;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class LIMEConverter {
	public static List<IPattern> getAllNominal(String file, Instances instances){
		List<IPattern> ps = new ArrayList<>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			for(Instance ins:instances){
				String line = br.readLine();
				IPattern pattern = parsePattern(line,ins);
				ps.add(pattern);
			}
			br.close();
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		return ps;
	}

	private static IPattern parsePattern(String line, Instance ins) {
		String[] tokens = line.split(",");
		Set<ICondition> conds = new HashSet<>();
		for(String token:tokens){
			int att = Integer.parseInt(token.trim());
			String value = ins.stringValue(att);
			conds.add(new NominalCondition(ins.attribute(att).name(), att, value));
		}
		return new Pattern(conds);
	}
	
	public static void main(String[] args) throws Exception{
		Instances data = DataUtils.load("data/synthetic/balloon_noisy_test.arff");
		System.out.println(getAllNominal("tmp/limeRes",data));
	}
}
