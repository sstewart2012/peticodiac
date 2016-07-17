#ifndef cudaHelper_h
#define cudaHelper_h

#include <iostream>
#include <cstdio>
#include <sys/time.h>

static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
#define CHECK(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define memcpyH2D(dst, src, n) (CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)))
#define memcpyD2H(dst, src, n) (CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost)))
#define memcpyD2D(dst, src, n) (CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice)))
#define memfree(d_ptr) (CHECK(cudaFree(d_ptr)))
#define memalloc(ptr, n) (CHECK(cudaMalloc(ptr, n)))

#endif
