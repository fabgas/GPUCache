/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */
extern "C"
__global__ void sumfloat(float *g_idata,float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[]; 
    unsigned int tid = threadIdx.x; // thread courant dans le block
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // index général
	sdata[tid] = g_idata[i]; // copy vers la shared memory du block
	__syncthreads(); // on attends tous les blocks
	
	if (i >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) { // si correspond à un multiple de la dimension
		sdata[tid] += sdata[tid + s];
		}
		__syncthreads(); // on attends
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

extern "C"
__global__ void minfloat(float *g_idata,float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[]; 
    unsigned int tid = threadIdx.x; // thread courant dans le block
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // index général
	sdata[tid] = g_idata[i]; // copy vers la shared memory du block
	__syncthreads(); // on attends tous les blocks
	
	if (i >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) { // si correspond à un multiple de la dimension
			if (sdata[tid+s]<sdata[tid]) {
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads(); // on attends
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

extern "C"
__global__ void maxfloat(float *g_idata,float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[]; 
    unsigned int tid = threadIdx.x; // thread courant dans le block
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // index général
	sdata[tid] = g_idata[i]; // copy vers la shared memory du block
	__syncthreads(); // on attends tous les blocks
	
	if (i >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) { // si correspond à un multiple de la dimension
			if (sdata[tid+s]>sdata[tid]) {
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads(); // on attends
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}
extern "C" 
__global__ void stringfilter(char2 *text,unsigned int char_per_word,unsigned int n) 
{
	unsigned int id = threadIdx.x; // numero de thread courant
	unsigned int index = blockIdx.x * blockDim.x + id; // index absolu
	
	//recherche du texte
	unsigned int offset = index * char_per_word;
	if (index <n) {
		printf(" texte %c %d",text[offset],index);
	}
	
}