#include<stdio.h>

#define N 20
#define THREADS_PER_BLOCK 5
#define RADIUS 3
#define BLOCK_SIZE N/THREADS_PER_BLOCK


__global__ void stencil_ld(int *in, int *out){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if(threadIdx.x<RADIUS){
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = 
        in[gindex + BLOCK_SIZE];
    }

    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += temp[lindex + offset];
    }

    out[gindex] = result;
}

int main(void){
    int *in, *out;
    int *d_in, *d_out;
    int size = N * sizeof(int);

    cudaMalloc((void **)&d_in,size);
    cudaMalloc((void **)&d_out,size);

    in = (int *)malloc(size);

    for (int i = 0; i<N; i++){
        in[i]=i;
    }

    out = (int *)malloc(size); 

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
 
    stencil_ld<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_in,d_out);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; i++){
        printf("%d ",in[i]);
    }
    printf("\n");

    for (int i = 0; i<N; i++){
        printf("%d ",out[i]);
    }

    printf("\n");

    free(in);
    free(out);

    // CleanUp
    cudaFree(d_in); 
    cudaFree(d_out);

}