#include<stdio.h>

__global__ void add(int *a, int *b, int *c){
    printf("testxxxxx %d\n",c[blockIdx.x]);
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    printf("test");
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = (rand() % 40);
}

#define N 512
int main(void){
    printf("Hello World!\n");

    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);

    // Allocate spaces for host copies for a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a,N);
    b = (int *)malloc(size); random_ints(b,N);
    c = (int *)malloc(size); 
    
    // Setup input values
    //a = 2;
    //b = 7;
    
    // Copy inputs to devices
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    printf("test point 1 \n");
    // Launch add() kernel on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    printf("test point 2 \n");
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    
    for (int i = 0; i<N; i++){
        printf("i = %d , a = %d , b = %d , c = %d \n",i,a[i],b[i],c[i]);
    }
    
    free(a);
    free(b);
    free(c);

    // CleanUp
    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
