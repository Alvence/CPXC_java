package com.yunzhejia.globallocal.classifier;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;

import com.yunzhejia.cpxc.util.ClassifierGenerator;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassifierIllustration {

	public static void main(String[] args) {
		try {
			DataSource source;
			Instances data;
	
			source = new DataSource("data/synthetic2.arff");
			data = source.getDataSet();

			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(GreedyGlobalLocalClassifier.globalType);
			cl.buildClassifier(data);
			
			 Writer writer = new BufferedWriter(new OutputStreamWriter(
		              new FileOutputStream("tmp/res2"), "UTF-8"));
			Writer writer2 = new BufferedWriter(new OutputStreamWriter(
		              new FileOutputStream("tmp/res"), "UTF-8"));
//		    for (double x = 0; x < 20; x+=0.1){
//		    	for(double y = -10; y < 45; y+=0.1){
//		    		Instance newIns = data.firstInstance();
//		    		newIns.setValue(0, x);
//		    		newIns.setValue(1, y);
//		    		if (cl.classifyInstance(newIns)==0){
//		    			writer2.write(x+","+y + ","+ cl.distributionForInstance(newIns)[0] +"\n");
//		    		}else{
//		    			writer.write(x+","+y + ","+ cl.distributionForInstance(newIns)[1] +"\n");
//		    		}
//		    	}
//		    }
		    
		    for(Instance ins:data){
		    	double err = 0;
		    	if (cl.classifyInstance(ins)==ins.classValue()){
		    		err = 1 - cl.distributionForInstance(ins)[(int)ins.classValue()];
		    	}else{
		    		err = 1.0;
		    	}
	    		writer2.write(ins.value(0)+","+ins.value(1) + ","+ err +"\n");
	    		
		    }
		    
		    writer2.close();
		    writer.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}

}
