package com.yunzhejia.cpxc.pattern;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.yunzhejia.cpxc.Discretizer;

import weka.core.Instances;

public class PatternSet extends AbstractList<Pattern>{
	private List<Pattern> patterns;
	
	public PatternSet(){
		patterns = new ArrayList<Pattern>();
	}
	
	public PatternSet(PatternSet copy){
		patterns = new ArrayList<Pattern>(copy.patterns);
	}

	public void readPatterns(String filename){
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
			while(line != null){
				Pattern pattern = Pattern.parsePattern(line);
				if(pattern!=null){
					patterns.add(pattern);
				}
				line = br.readLine();
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void contrast(Instances LE, Instances SE, Discretizer discretizer, double minRatio) {
		Iterator<Pattern> it = patterns.iterator();
		while(it.hasNext()){
			Pattern p = it.next();
			float supLE = p.supportOfData(LE, discretizer);
			float supSE = p.supportOfData(SE, discretizer);
			if (supLE/supSE < 2){
				it.remove();
			}
		}
	}
	

	@Override
	public Pattern get(int index) {
		return patterns.get(index);
	}

	@Override
	public int size() {
		return patterns.size();
	}
	
	@Override
	public void add(int index, Pattern pattern){
		patterns.add(index, pattern);
	}
	
	@Override
	public Pattern set(int index, Pattern pattern){
		Pattern oldPattern = patterns.get(index);
		patterns.set(index, pattern);
		return oldPattern;
	}
	
	@Override
	public Pattern remove(int index){
		Pattern oldPattern = patterns.get(index);
		patterns.remove(index);
		return oldPattern;
	}
	
}
