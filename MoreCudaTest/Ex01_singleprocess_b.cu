/* 
 * COMPILATION TIP
 * nvcc -std=c++11 -I ~/nccl/include -L ~/nccl/lib -lnccl nccl_playground.cu -o nccl_playground.exe
 * */
#include <iostream> // std::cout
#include <memory> 	// std::unique_ptr
#include <vector> 	// std::vector

#include "nccl.h"

int main(int argc, char* argv[])
{
	// managing 1 device
	auto comm_deleter=[&](ncclComm_t* comm){ ncclCommDestroy( *comm ); };
	std::unique_ptr<ncclComm_t, decltype(comm_deleter)> comm(new ncclComm_t, comm_deleter);	


	// sanity check of number of GPUs 
	int nDev = 0;
	cudaGetDeviceCount(&nDev);
	if (nDev == 0) {
		std::cout << "No GPUs found " << std::endl; 
		exit(EXIT_FAILURE);
	}
	std::cout << " nDev (number of devices) : " << nDev << std::endl;
	// END of sanity check of number of GPUs 


	int size = 32 * 1024 * 1024;
	std::cout << " size : " << size << std::endl; 
	int devs[1] = {0};
	
	// generate input vector/array on host
	std::vector<float> f_vec(size,2.f);
	

	// device pointers
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> d_in(new float[size], deleter);
	cudaMalloc((void **) &d_in, size * sizeof(float));

	std::unique_ptr<float[], decltype(deleter)> d_out(new float[size], deleter);
	cudaMalloc((void **) &d_out, size * sizeof(float));


	// CUDA stream smart pointer stream
	auto stream_deleter=[&](cudaStream_t* stream){ cudaStreamDestroy( *stream ); };
	std::unique_ptr<cudaStream_t, decltype(stream_deleter)> stream(new cudaStream_t, stream_deleter);
	cudaStreamCreate(stream.get());
	

	cudaMemcpy(d_in.get(), f_vec.data(), size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemset(d_out.get(), 0.f, size*sizeof(float));

	cudaMemcpy(f_vec.data(), d_out.get(), size*sizeof(float), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();

	//initializing NCCL
	ncclCommInitAll(comm.get(), nDev, devs);

	// number of ranks in a communicator
	int count =0;
	ncclCommCount(*comm.get(),&count);
	std::cout << " number of ranks in a communicator, using ncclCommCount : " << count << std::endl;

	ncclAllReduce( d_in.get(), d_out.get(), size, ncclFloat, ncclSum, *comm.get(), *stream.get() );


	// read out output in host
	cudaMemcpy(f_vec.data(), d_out.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "On the device:  f_vec[0] : " << f_vec[0] << ", f_vec[1] : " << f_vec[1] << 
		", f_vec[2] : " << f_vec[2] << ", f_vec[3] : " << f_vec[3] << 
		", f_vec.back() : " << f_vec.back() << std::endl;
	for (int idx=4; idx < 32+4 ; idx++) {
		std::cout << idx << " : " << f_vec[idx] << " " ; 
	}


	cudaDeviceReset();
		
	return 0;
}