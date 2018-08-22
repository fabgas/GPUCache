package org.gpucache.cache;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

public class GPUUtil {
	
	private CUmodule module;
	private CUfunction functionSumFloat;
	private CUfunction functionMinFloat;
	private CUfunction functionMaxFloat;
	private CUfunction functionFilterString;
	private static GPUUtil instance;
	
	private GPUUtil() {
		super();
		initContext();
	}

	public static GPUUtil getInstance() {
		if (instance == null) {
			instance = new GPUUtil();
		}
		return instance;
	}
	
	 private  CUcontext initContext()
	    {
	        cuInit(0);
	        CUdevice device = new CUdevice();
	        cuDeviceGet(device, 0);
	        CUcontext  context = new CUcontext();
	        cuCtxCreate(context, 0, device);
	        prepare();
	        return context;
	    }
	    /**
	     * Prepare everything for calling the reduction kernel function.
	     * This method assumes that a context already has been created
	     * and is current!
	     */
	 private  void prepare()
	    {
	        // Prepare the ptx file.
	        String ptxFileName = null;
	        try
	        {
	            ptxFileName = preparePtxFile("operations.cu");
	        }
	        catch (IOException e)
	        {
	            throw new RuntimeException("Could not prepare PTX file", e);
	        }
	        
	        // Load the module from the PTX file
	        module = new CUmodule();
	        cuModuleLoad(module, ptxFileName);

	        // Obtain a function pointer to the "reduce" function.
	        functionSumFloat = new CUfunction();
	        cuModuleGetFunction(functionSumFloat, module, "sumfloat");
	        functionMinFloat= new CUfunction();
	        cuModuleGetFunction(functionMinFloat, module, "minfloat");
	        functionMaxFloat= new CUfunction();
	        cuModuleGetFunction(functionMaxFloat, module, "maxfloat");
	        functionFilterString= new CUfunction();
	        cuModuleGetFunction(functionFilterString, module, "stringfilter");
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
	    public static byte[] toByteArray(InputStream inputStream)
	        throws IOException
	    {
	        ByteArrayOutputStream baos = new ByteArrayOutputStream();
	        byte buffer[] = new byte[8192];
	        while (true)
	        {
	            int read = inputStream.read(buffer);
	            if (read == -1) break;
	            
	            baos.write(buffer, 0, read);
	        }
	        return baos.toByteArray();
	    }

		public CUmodule getModule() {
			return module;
		}

		public void setModule(CUmodule module) {
			this.module = module;
		}

		public CUfunction getFunction(Operation operation) {
			if (Operation.SUM==operation) {
				return functionSumFloat;
			}
			if (Operation.MAX==operation) {
				return functionMaxFloat;
			}
			if (Operation.FILTER==operation) {
				return functionFilterString;
			}
			return functionMinFloat;
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
		public static int getNumBlocks(int n, int maxBlocks, int maxThreads)
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
	    public static int getNumThreads(int n, int maxBlocks, int maxThreads)
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
	    public static int nextPow2(int x)
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
