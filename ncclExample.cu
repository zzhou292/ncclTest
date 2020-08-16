#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>


// an cuda cmd check from Nvidia website
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// an nccl cmd check from Nvidia website
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)




int main(int argc, char* argv[])
{
  ncclComm_t comms[4];


  //managing 4 devices
  int nDev = 4;
  int size = 32*1024; //allocate 32K
  int devs[4] = { 0, 1, 2, 3 };


  //allocating and initializing device buffers
  //malloc space on CPU
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  //setting up sendbuff and receive buff
  //setting array values in sendbuff and recv buff
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    //CUDACHECK(cudaMemset(sendbuff[i], 0, size * sizeof(float)));  // default value set from nccl website IGNORE!
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));

    float sendValue = 0.;
    for(int j = 0; j < size ; j++){
      if(sendValue>100.){sendValue = 0.;}
      //set send value as 0.0 1.0 2.0 3.0 .....100.0 0.0 1.0 2.0......
      CUDACHECK(cudaMemset(sendbuff[i]+j, sendValue, sizeof(float)));
      sendValue = sendValue + 1.0;
    }

    CUDACHECK(cudaStreamCreate(s+i));
  }




  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //display testing result
  //not sure about the result
  for (int i = 0; i < nDev ; ++i){
    for(int j = 0; j < size; j++){
      std::cout<<"i: "<<i<<"j: "<<j<<"  "<<recvbuff[i]<<std::endl;
    }
    
  }


  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  //display success msg
  printf("Test program finished \n");
  return 0;
}
