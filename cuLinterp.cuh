#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

// print head elements of an array
#define PRINT_HEAD_1D(Data, Head) for (int i = 0; i < Head; ++i) i == 0 ? printf(#Data"[0-%d...]: %.4f, ", Head-1, Data[0]): ( i == Head-1 ? printf("%.4f...\n", Data[i]) : printf("%.4f, ", Data[i]))
#define PRINT_HEAD_ROWWISE_2D(Data, Row, Col, Head1, Head2) for (int i = 0; i < Head1; ++i) for (int j = 0; j < Head2; ++j) j == 0 ? printf(#Data"[%d, 0-%d...]: %.4f, ", i, Head2-1, Data[i*Col]): ( j == Head2 - 1 ? printf("%.4f...\n", Data[i*Col+j]) : printf("%.4f, ", Data[i*Col+j]))

// cuda-error detection function
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t err, const char* const func, const char* const file, int const line)
{
	if (err != cudaSuccess) 
		fprintf(stderr, "CUDA Runtime Error at: %s: %s\n %s %s\n", file, line, cudaGetErrorString(err), func);
}

// cufft-error detection function
#define CHECK_CUFFT_ERROR(val) checkCufft((val), #val, __FILE__, __LINE__)
void checkCuFFT(cufftResult_t err, const char* const func, const char* const file, int const line)
{
	if (err != CUFFT_SUCCESS) 
		fprintf(stderr, "cuFFT Runtime Error at: %s: %s\n %s\n", file, line, func);
}

// cuda-last-error detection function
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
	cudaError_t const err{ cudaGetLastError() };
	if (err != cudaSuccess)
		fprintf(stderr, "CUDA Runtime Error at: %s: %s\n %s \n", file, line, cudaGetErrorString(err));
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

// float API for 1d linear interpolation running on GPUs 
void fcuLinterp1d(float* fpOutput, const float* const fpInput,
const float* const sampleXgrid, const int size_xs,
const float* const interpXgrid, const int size_xi)
{
	// TODO...
}

// float API for 2d linear interpolation running on NVIDIA GPUs through CUDA 
void fcuLinterp2d(
	float* fpOutput, const float* const fpInput,
	const float* const sampleXgrid, const int size_xs,
	const float* const sampleYgrid, const int size_ys,
	const float* const interpXgrid, const int size_xi, 
	const float* const interpYgrid, const int size_yi)
{
	/*
	----------> x(column)  ----------> x(column)
	|                      |
	| SampleGrid   =====>> | InterpolationGrid
	|                      |
	y(row)                 y(row)
	*/
	float* d_ygrid_sample; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ygrid_sample, sizeof(float) * size_ys));
	float* d_xgrid_sample; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xgrid_sample, sizeof(float) * size_xs));
	float* d_ygrid_interp; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ygrid_interp, sizeof(float) * size_yi));
	float* d_xgrid_interp; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xgrid_interp, sizeof(float) * size_xi));
	float* d_fpOutput; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_fpOutput, sizeof(float) * size_xi * size_yi));
	float* d_fpInput; CHECK_CUDA_ERROR(cudaMalloc((void**)&d_fpInput, sizeof(float) * size_xs * size_ys));
	CHECK_CUDA_ERROR(cudaMemcpy(d_ygrid_sample, sampleYgrid, sizeof(float) * size_ys, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_xgrid_sample, sampleXgrid, sizeof(float) * size_xs, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_ygrid_interp, interpYgrid, sizeof(float) * size_yi, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_xgrid_interp, interpXgrid, sizeof(float) * size_xi, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_fpInput, fpInput, sizeof(float) * size_xs * size_ys, cudaMemcpyHostToDevice));

	dim3 blockDim(16, 16, 1);
	dim3 gridDim((size_xi + blockDim.x - 1) / blockDim.x, (size_yi + blockDim.y - 1) / blockDim.y, 1);
	interp2dKernel << <gridDim, blockDim >> > (d_fpOutput, d_fpInput, d_xgrid_sample, size_xs, d_ygrid_sample, size_ys, d_xgrid_interp, size_xi, d_ygrid_interp, size_yi);
	CHECK_LAST_CUDA_ERROR();
	CHECK_CUDA_ERROR(cudaMemcpy(fpOutput, d_fpOutput, sizeof(float) * size_xi * size_yi, cudaMemcpyDeviceToHost));

	cudaFree(d_ygrid_sample);
	cudaFree(d_xgrid_sample);
	cudaFree(d_ygrid_interp);
	cudaFree(d_xgrid_interp);
	cudaFree(d_fpOutput);
	cudaFree(d_fpInput);

	return;
}

// float API for 1d linear interpolation running on CPUs
void fLinterp1d(
	float* fpOutput, const float* const fpInput,
	const float* const sampleXgrid, const int size_xs,
	const float* const interpXgrid, const int size_xi)
{   
	// Sample x Grid =====>> Interpolation x grid  

	// We assume that all grids are even! 
	// i.e. This function only works for even grids;
	float step_xs = sampleXgrid[1] - sampleXgrid[0];
	int xi,  xi_1;
	float a, b;
	for (int i{ 0 }; i < size_xi; ++i) 
	{
		if (interpXgrid[i] < sampleXgrid[0] || interpXgrid[i] > sampleXgrid[size_xs - 1])
		{
			fpOutput[i] = 0.0f;
			continue;
		}

		xi = floor((interpXgrid[i] - sampleXgrid[0]) / step_xs);
		xi_1 = xi + 1;
		a = (fpInput[xi_1] - fpInput[xi]) / step_xs;
		b = fpInput[xi] - a * sampleXgrid[xi];
		fpOutput[i] = a * interpXgrid[i] + b;
	}
}

// float API for 2d linear interpolation running on CPUs
void fLinterp2d(
	float* fpOutput, const float* const fpInput,
	const float* const sampleXgrid, const int size_xs,
	const float* const sampleYgrid, const int size_ys,
	const float* const interpXgrid, const int size_xi,
	const float* const interpYgrid, const int size_yi)
{
	/*
	----------> x(column)  ----------> x(column)
	|                      |
	| SampleGrid   =====>> | InterpolationGrid
	|                      |
	y(row)                 y(row)
	*/

	// We assume that all grids are even! 
	// i.e. This function only works for even grids;
	float step_ys = sampleYgrid[1] - sampleYgrid[0];
	float step_xs = sampleXgrid[1] - sampleXgrid[0];

	int yi, xj, yi_1, xj_1;
	float u, v, z, t;

	for (int i{ 0 }; i < size_yi; ++i)
	{
		for (int j{ 0 }; j < size_xi; ++j)
		{
			if (interpYgrid[i] < sampleYgrid[0] || interpYgrid[i] > sampleYgrid[size_ys - 1] ||
				interpXgrid[j] < sampleXgrid[0] || interpXgrid[j] > sampleXgrid[size_xs - 1])
			{
				fpOutput[i * size_xi + j] = 0.0f;
				continue;
			}

			yi = floor((interpYgrid[i] - sampleYgrid[0]) / step_ys);
			xj = floor((interpXgrid[j] - sampleXgrid[0]) / step_xs);
			yi_1 = yi + 1;
			xj_1 = xj + 1;

			//			printf("i, j = %d, %d\n", i, j);
			//			printf("yi, xj = %d, %d\n", yi, xj);

			if (yi == size_ys - 1) {
				v = 0;
				yi_1 = 0; // verbose
			}
			else {
				v = (interpYgrid[i] - sampleYgrid[yi]) / (sampleYgrid[yi_1] - sampleYgrid[yi]);
			}

			if (xj == size_xs - 1) {
				u = 0;
				xj_1 = 0; // verbose
			}
			else {
				u = (interpXgrid[j] - sampleXgrid[xj]) / (sampleXgrid[xj_1] - sampleXgrid[xj]);
			}

			t = fpInput[yi * size_xs + xj];
			if (v <= u) {
				z = t + u * (fpInput[yi * size_xs + xj_1] - t) + v * (fpInput[yi_1 * size_xs + xj_1] - fpInput[yi * size_xs + xj_1]);
			}
			else {
				z = t + v * (fpInput[yi_1 * size_xs + xj] - t) + u * (fpInput[yi_1 * size_xs + xj_1] - fpInput[yi_1 * size_xs + xj]);
			}

			fpOutput[i * size_xi + j] = z;
		}
	}

	return;
}
