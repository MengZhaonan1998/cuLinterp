#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

// cuda error-detection function
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, int const line)
{
	if (err != cudaSuccess) {
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// cuda error-detection function
void check(cufftResult_t err, const char* const func, const char* const file, int const line)
{
	if (err != CUFFT_SUCCESS) {
		std::cerr << "CUFFT Error " << err << " at: " << file << ":" << line << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// cuda error-detection function
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
	cudaError_t const err{ cudaGetLastError() };
	if (err != cudaSuccess) {
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define print_arrayElement(h_array, idx) std::cout << #h_array << "["<< idx << "] = " << h_array[idx] << "\n"
template <class T>
void printArray1dHead(T* h_array, int head)
{
	for (int i = 0; i < head; i++)
		print_arrayElement(h_array, i);
}

__global__ void interp2dKernel(
	float* d_fpOutput,
	const float* const d_fpInput,
	const float* const d_sampleXgrid, int size_xs,
	const float* const d_sampleYgrid, int size_ys,
	const float* const d_interpXgrid, int size_xi,
	const float* const d_interpYgrid, int size_yi)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	if (j >= size_xi) return;
	if (i >= size_yi) return;

	// We assume that all grids are even! 
	// i.e. This function only works for even grids;
	float step_ys = d_sampleYgrid[1] - d_sampleYgrid[0];
	float step_xs = d_sampleXgrid[1] - d_sampleXgrid[0];
	int yi, xj, yi_1, xj_1;
	float u, v, z, t;

	if (d_interpYgrid[i] < d_sampleYgrid[0] || d_interpYgrid[i] > d_sampleYgrid[size_ys - 1] ||
		d_interpXgrid[j] < d_sampleXgrid[0] || d_interpXgrid[j] > d_sampleXgrid[size_xs - 1])
	{
		d_fpOutput[i * size_xi + j] = 0.0f;
		return;
	}

	yi = floor((d_interpYgrid[i] - d_sampleYgrid[0]) / step_ys);
	xj = floor((d_interpXgrid[j] - d_sampleXgrid[0]) / step_xs);
	yi_1 = yi + 1;
	xj_1 = xj + 1;
	//printf("i, j = %d, %d\n", i, j);
	//printf("yi, xj = %d, %d\n", yi, xj);

	if (yi == size_ys - 1) {
		v = 0;
		yi_1 = 0; // verbose
	}
	else {
		v = (d_interpYgrid[i] - d_sampleYgrid[yi]) / (d_sampleYgrid[yi_1] - d_sampleYgrid[yi]);
	}

	if (xj == size_xs - 1) {
		u = 0;
		xj_1 = 0; // verbose
	}
	else {
		u = (d_interpXgrid[j] - d_sampleXgrid[xj]) / (d_sampleXgrid[xj_1] - d_sampleXgrid[xj]);
	}

	t = d_fpInput[yi * size_xs + xj];
	if (v <= u) {
		z = t + u * (d_fpInput[yi * size_xs + xj_1] - t) + v * (d_fpInput[yi_1 * size_xs + xj_1] - d_fpInput[yi * size_xs + xj_1]);
	}
	else {
		z = t + v * (d_fpInput[yi_1 * size_xs + xj] - t) + u * (d_fpInput[yi_1 * size_xs + xj_1] - d_fpInput[yi_1 * size_xs + xj]);
	}

	d_fpOutput[i * size_xi + j] = z;
	return;
}


void interp2d_gpu(
	float* fpOutput, const float* const fpInput,
	const std::vector<float> sampleXgrid, const std::vector<float> sampleYgrid,
	const std::vector<float> interpXgrid, const std::vector<float> interpYgrid)
{
	int size_ys = sampleYgrid.size();
	int size_xs = sampleXgrid.size();
	int size_yi = interpYgrid.size();
	int size_xi = interpXgrid.size();
	float* d_ygrid_sample; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ygrid_sample, sizeof(float) * size_ys));
	float* d_xgrid_sample; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xgrid_sample, sizeof(float) * size_xs));
	float* d_ygrid_interp; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ygrid_interp, sizeof(float) * size_yi));
	float* d_xgrid_interp; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xgrid_interp, sizeof(float) * size_xi));
	float* d_fpOutput; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_fpOutput, sizeof(float) * size_xi * size_yi));
	float* d_fpInput; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_fpInput, sizeof(float) * size_xs * size_ys));
	CHECK_CUDA_ERROR(cudaMemcpy(d_ygrid_sample, sampleYgrid.data(), sizeof(float) * size_ys, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_xgrid_sample, sampleXgrid.data(), sizeof(float) * size_xs, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_ygrid_interp, interpYgrid.data(), sizeof(float) * size_yi, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_xgrid_interp, interpXgrid.data(), sizeof(float) * size_xi, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_fpInput, fpInput, sizeof(float) * size_xs * size_ys, cudaMemcpyHostToDevice));

	dim3 blockDim(16, 16, 1);
	dim3 gridDim((size_xi + blockDim.x - 1) / blockDim.x, (size_yi + blockDim.y - 1) / blockDim.y, 1);

	interp2dKernel << <gridDim, blockDim >> > (d_fpOutput, d_fpInput, d_xgrid_sample, size_xs, d_ygrid_sample, size_ys, d_xgrid_interp, size_xi, d_ygrid_interp, size_yi);
	//std::cout << mIter << " iterations of interp2dKernel took " << (end_t - start_t) / CLOCKS_PER_SEC << " seconds." << std::endl;

	CHECK_CUDA_ERROR(cudaMemcpy(fpOutput, d_fpOutput, sizeof(float) * size_xi * size_yi, cudaMemcpyDeviceToHost));

	cudaFree(d_ygrid_sample);
	cudaFree(d_xgrid_sample);
	cudaFree(d_ygrid_interp);
	cudaFree(d_xgrid_interp);
	cudaFree(d_fpOutput);
	cudaFree(d_fpInput);

	return;
}
