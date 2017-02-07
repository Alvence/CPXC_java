package com.yunzhejia.cpxc.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;

public class DataUtils {
	 /**
	    * loads the given ARFF file and sets the class attribute as the last
	    * attribute.
	    *
	    * @param filename    the file to load
	    * @throws Exception  if somethings goes wrong
	    */
	   public static Instances load(String filename) throws Exception {
	     Instances       result;
	     BufferedReader  reader;
	 
	     reader = new BufferedReader(new FileReader(filename));
	     result = new Instances(reader);
	     result.setClassIndex(result.numAttributes() - 1);
	     reader.close();
	 
	     return result;
	   }
	/**
	    * saves the data to the specified file
	    *
	    * @param data        the data to save to a file
	    * @param filename    the file to save the data to
	    * @throws Exception  if something goes wrong
	    */
	   public static void save(Instances data, String filename) throws Exception {
	     BufferedWriter  writer;
	 
	     writer = new BufferedWriter(new FileWriter(filename));
	     writer.write(data.toString());
	     writer.newLine();
	     writer.flush();
	     writer.close();
	   }
	  
}
