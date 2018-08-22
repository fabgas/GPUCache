package org.gpucache.cache;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static org.gpucache.cache.GPUUtil.getNumBlocks;
import static org.gpucache.cache.GPUUtil.getNumThreads;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;;
public class GPUOperation {
	/**
	 * Do the sum of n elements
	 * @param elements
	 * @return
	 */
	public static Float sum(float[] elements) {
		// calculate the number of threads
		int length = elements.length;
		int blocks = getNumBlocks(length, 50000, 512)*2;
		int threads = getNumThreads(length, 50000*16, 512);
		// instanciate buffer
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, length * Sizeof.FLOAT);
		cuMemcpyHtoD(deviceInput, Pointer.to(elements), 
				length * Sizeof.FLOAT);

		// Allocate a chunk of temporary memory (must be at least
		// numberOfBlocks * Sizeof.FLOAT)
		CUdeviceptr deviceBuffer = new CUdeviceptr();
		cuMemAlloc(deviceBuffer, blocks * Sizeof.FLOAT);
		Pointer kernelParameters = Pointer.to(
				Pointer.to(deviceInput),
				Pointer.to(deviceBuffer),
				Pointer.to(new int[]{length})
				);

		kernel(GPUUtil.getInstance().getFunction(Operation.SUM),kernelParameters,length,blocks,threads);

		float result[] = new float[blocks];
		cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT*blocks);
		float sum = 0.0f;
		for (int i = 0; i < result.length; i++) {
			sum +=result[i];
		}
		return sum;
	}

	public static Float min(float[] elements) {
		// calculate the number of threads
		int length = elements.length;
		int blocks = getNumBlocks(length, 50000, 512)*2;
		int threads = getNumThreads(length, 50000*16, 512);
		// instanciate buffer
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, length * Sizeof.FLOAT);
		cuMemcpyHtoD(deviceInput, Pointer.to(elements), 
				length * Sizeof.FLOAT);

		// Allocate a chunk of temporary memory (must be at least
		// numberOfBlocks * Sizeof.FLOAT)
		CUdeviceptr deviceBuffer = new CUdeviceptr();
		cuMemAlloc(deviceBuffer, blocks * Sizeof.FLOAT);
		Pointer kernelParameters = Pointer.to(
				Pointer.to(deviceInput),
				Pointer.to(deviceBuffer),
				Pointer.to(new int[]{length})
				);

		kernel(GPUUtil.getInstance().getFunction(Operation.MIN),kernelParameters,length,blocks,threads);

		float result[] = new float[blocks];
		cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT*blocks);
		float min = result[0];
		
		for (int i = 1; i < result.length; i++) {
			if(result[i]<min) min = result[i];
		}
		return min;
	}

	public static Float max(float[] elements) {
		// calculate the number of threads
		int length = elements.length;
		int blocks = getNumBlocks(length, 50000, 512)*2;
		int threads = getNumThreads(length, 50000*16, 512);
		// instanciate buffer
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, length * Sizeof.FLOAT);
		cuMemcpyHtoD(deviceInput, Pointer.to(elements), 
				length * Sizeof.FLOAT);

		// Allocate a chunk of temporary memory (must be at least
		// numberOfBlocks * Sizeof.FLOAT)
		CUdeviceptr deviceBuffer = new CUdeviceptr();
		cuMemAlloc(deviceBuffer, blocks * Sizeof.FLOAT);
		Pointer kernelParameters = Pointer.to(
				Pointer.to(deviceInput),
				Pointer.to(deviceBuffer),
				Pointer.to(new int[]{length})
				);

		kernel(GPUUtil.getInstance().getFunction(Operation.MAX),kernelParameters,length,blocks,threads);

		float result[] = new float[blocks];
		cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT*blocks);
		float max = result[0];
		
		for (int i = 1; i < result.length; i++) {
			if(result[i]>max) max = result[i];
		}
		return max;
	}
	public static void filter(char[]  elements,int maxsize) {
		int length = elements.length;
		int blocks = getNumBlocks(length, 50000, 512)*2;
		int threads = getNumThreads(length, 50000*16, 512);
		// instanciate buffer
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, length*2 );
		cuMemcpyHtoD(deviceInput, Pointer.to(elements), length*2);

		// Allocate a chunk of temporary memory (must be at least
		// numberOfBlocks * Sizeof.FLOAT)
		CUdeviceptr deviceBuffer = new CUdeviceptr();
		cuMemAlloc(deviceBuffer, blocks * Sizeof.CHAR);
		Pointer kernelParameters = Pointer.to(
				Pointer.to(deviceInput),
				Pointer.to(new int[]{maxsize}), // shift for each string
				Pointer.to(new int[]{length})
				);
		// parametres : byte:[], maxsize of a string (need termination ?)
		kernel(GPUUtil.getInstance().getFunction(Operation.FILTER),kernelParameters,length,blocks,threads);

	
	}
	private static  void kernel(CUfunction function, Pointer kernelParameters, int length,int blocks,int threads) {

		int sharedMemSize = threads * Sizeof.FLOAT;
		// Call the kernel function.
		cuLaunchKernel(function,
				blocks,  1, 1,         // Grid dimension
				threads, 1, 1,         // Block dimension
				sharedMemSize, null,   // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
				);
		cuCtxSynchronize();

	}
}
