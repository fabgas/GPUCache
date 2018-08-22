package test;

import org.gpucache.cache.GPUOperation;
import org.gpucache.cache.GPUUtil;

public class ContextTest {

	public static void main(String[] args) {
		// GPUUtil
		GPUUtil.getInstance();
		int size = 1<<24;
		System.out.println(size);
		float[] array = new float[size];
		for(int i = 0; i < size; i++)
		{
			array[i] =i*1.0f;
		}
		System.out.println(GPUOperation.sum(array));
		System.out.println(GPUOperation.min(array));
		System.out.println(GPUOperation.max(array));
		char[] chars = new char[10];
		chars[0]="a".charAt(0);
		chars[1]="é".charAt(0);
		chars[2]="è".charAt(0);
		chars[3]="à".charAt(0);
		chars[4]="c".charAt(0);
		chars[5]="d".charAt(0);
		chars[6]="7".charAt(0);
		chars[7]="ù".charAt(0);
		chars[8]="f".charAt(0);
		chars[9]="v".charAt(0);
		GPUOperation.filter(chars, 1);
	}

}
