package com.yunzhejia.pattern.patternmining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class GcGrowthPatternMiner implements IPatternMiner {

	protected transient Discretizer discretizer;
	protected transient Instances data;
	
	public GcGrowthPatternMiner(Discretizer discretizer) {
		this.discretizer = discretizer;
	}


	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public PatternSet minePattern(Instances data,  double minSupp){
		this.data = data;
		PatternSet ps = null;
		String tmpFile = "tmp/dataForPattern.txt";
		String patternFile = "tmp/output.key";
		File file = new File(tmpFile);
		try {
			PrintWriter writer = new PrintWriter(file);
			for(int i = 0; i < data.numInstances(); i++){
				Instance ins = data.get(i);
				writer.println(discretizer.getDiscretizedInstance(ins));
			}
			writer.close();
			
			String[] cmd = {"program\\GcGrowth.exe", tmpFile,(int)(minSupp*data.numInstances())+"","tmp\\output" };
			Process process = new ProcessBuilder(cmd).start();
			//wait until the program terminates
			while(isRunning(process)){}
			ps = new PatternSet();
			
			BufferedReader br = new BufferedReader(new FileReader(patternFile));
			String line = br.readLine();
			while(line != null){
				IPattern pattern = parsePattern(line);
				if(pattern!=null){
					ps.add(pattern);
				}
				line = br.readLine();
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		/*verify support for patterns
		for (Pattern p:ps.patterns){
			if(p.getSupport() != p.supportOfData(data, discretizer)){
				System.out.println("!!  "+p);
			}
		}
		*/
		return ps;
	}

	private IPattern parsePattern(String str){
		String[] tokens = str.split(" ");
		if(tokens.length==2){
			return null;
		}
		Set<ICondition> conditions = new HashSet<>();
		for (int i = 0; i < tokens.length; i++){
			//ignore first
			if ( 0 == i){
				continue;
			}else if(tokens.length - 1 != i){ //conditions, or itemsets
				int disVal = Integer.parseInt(tokens[i]);
				int attrIndex = discretizer.getAttributeIndex(disVal);
				ICondition condition = null;
				String attrName = data.attribute(attrIndex).name();
				if (discretizer.isNumeric(attrIndex)){
					double left = discretizer.getLeft(disVal);
					double right = discretizer.getRight(disVal);
					condition = new NumericCondition(attrName,attrIndex, left, right);
				}else{
					String value = discretizer.getNominal(disVal);
					condition = new NominalCondition(attrName,i, value);
				}
				conditions.add(condition);
			}
		}
		IPattern pattern = new Pattern(conditions);
		return pattern;
	}
	
	private boolean isRunning(Process process) {
	    try {
	        process.exitValue();
	        return false;
	    } catch (Exception e) {
	        return true;
	    }
	}

public static void main(String[] args){
	try {
		DataSource source;
		Instances data;
		source = new DataSource("data/synthetic2.arff");
//		source = new DataSource("data/blood.arff");
//		source = new DataSource("data/iris.arff");
		data = source.getDataSet();
		
		if (data.classIndex() == -1){
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		
		 Discretizer discretizer = new Discretizer();
			discretizer.initialize(data);
			IPatternMiner patternMiner = new GcGrowthPatternMiner(discretizer);
			PatternSet ps = patternMiner.minePattern(data, 0.01);
			System.out.println(ps.size());
			for (IPattern p:ps){
				System.out.println(p);
			}
		
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
}

}
