package test;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class Sum {
	  /**
     * The CUDA context created by this sample
     */
    private static CUcontext context;
    /**
     * The module which is loaded in form of a PTX file
     */
    private static CUmodule module;
    
    /**
     * The actual kernel function from the module
     */
    private static CUfunction function;
    
    /**
     * Temporary memory for the device output
     */
    private static CUdeviceptr deviceBuffer;
    static int n = 1<<24;
	static int blocks = getNumBlocks(n, 50000, 512)*2;
  	static int threads = getNumThreads(n, 50000*16, 512); // threads par block ? limite 1024
	public static void main(String[] args) {
		 // Enable exceptions and omit all subsequent error checks
		System.out.println(" block :"+ blocks+ " / threads :"+ threads + " total :"+ blocks*threads );
        JCudaDriver.setExceptionsEnabled(true);
        init();
        //initialisation d'un pointeur pour la mémoire d'inout
        float hostInput[] = createRandomArray(blocks * threads);
        float sum = 0;
        for (int i =0; i< blocks* threads;i++) {
        	sum = sum + hostInput[i];
        }
        System.out.println(sum);
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, hostInput.length * Sizeof.FLOAT);
        long timecp0 = System.nanoTime();
        cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), 
            hostInput.length * Sizeof.FLOAT);
        long timecp1 = System.nanoTime();
        long durationcopy = timecp1 - timecp0;
        long time0 = System.nanoTime();
        float resultJCuda = helloworld(deviceInput, hostInput.length);
        long time1 = System.nanoTime();
        long durationCuda = time1 - time0;

        System.out.println("Reduction of " + n + " elements");
        System.out.printf("JCuda %5.3fms Copy %5.3fms \n",durationCuda/1e6,durationcopy/1e6);
//        System.out.printf(
//            "  JCuda: %5.3fms, result: %f " +
//            "(copy: %5.3fms, comp: %5.3fms)\n",
//            (durationCopy + durationComp) / 1e6, resultJCuda, 
//            durationCopy / 1e6, durationComp / 1e6);
        
        System.out.println("cuda" + resultJCuda);
        float tot = (blocks*1.0f*threads)*(blocks*1.0f*threads-1)/2.0f;
        System.out.println("Calcul exact :"+ tot);
        float percent = (resultJCuda-tot)/resultJCuda;
        System.out.println("Percent :"+ percent);
        
        cuMemFree(deviceInput);
	}
	
	public static float helloworld( Pointer deviceInput, int length) {
		  Pointer kernelParameters = Pointer.to(
		            Pointer.to(deviceInput),
		            Pointer.to(deviceBuffer),
		            Pointer.to(new int[]{length})
		        );
		 
		  
		    int sharedMemSize = threads * Sizeof.FLOAT;
		        // Call the kernel function.
		        cuLaunchKernel(function,
		            blocks,  1, 1,         // Grid dimension
		            threads, 1, 1,         // Block dimension
		            sharedMemSize, null,   // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
		        float result[] = new float[blocks];
		        cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT*blocks);     
		        float sum = 0.0f;
		        for (int i = 0; i < result.length; i++) {
					sum +=result[i];
				}
		        return sum;
		       
	}
	/**
     * Initialize the driver API and create a context for the first
     * device, and then call {@link #prepare()}
     */
    private static void init()
    {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        prepare();
    }
    
    /**
     * Prepare everything for calling the reduction kernel function.
     * This method assumes that a context already has been created
     * and is current!
     */
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxFileName = null;
        try
        {
            ptxFileName = preparePtxFile("sum.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "sum");
        
        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.FLOAT)
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, blocks * Sizeof.FLOAT);
        
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }
   
    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
    /**
     * Create an array of the given size, with random data
     * 
     * @param size The array size
     * @return The array
     */
    private static float[] createRandomArray(int size)
    {
        Random random = new Random(0);
        float array[] = new float[size];
        for(int i = 0; i < size; i++)
        {
            array[i] =i*1.0f;
        }
        return array;
    }
    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }
    
    /**
     * Returns the power of 2 that is equal to or greater than x
     * 
     * @param x The input
     * @return The next power of 2
     */
    private static int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }
}
