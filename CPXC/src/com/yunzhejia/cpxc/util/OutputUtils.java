package com.yunzhejia.cpxc.util;

import java.util.List;

public class OutputUtils {
	public static <T> void print(T[] elements){
		for (T element:elements){
			System.out.print(element+" ");
		}
		System.out.println();
	}
	
	public static <T> void print(List<T> elements){
		for (T element:elements){
			System.out.print(element+" ");
		}
		System.out.println();
	}

	public static void print(double[] elements) {
		for (double element:elements){
			System.out.print(element+" ");
		}
		System.out.println();
	}
	
	public static void print(int[] elements) {
		for (int element:elements){
			System.out.print(element+" ");
		}
		System.out.println();
	}
}
