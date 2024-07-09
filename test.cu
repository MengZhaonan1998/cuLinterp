#include "cuLinterp.cuh"

// return an evenly spaced 1-d grid of floats/doubles.
void linspacef(float* fArray,  float first, float last, int len) {
	double step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { fArray[i] = first + i * step; }
}

void linspaced(double* fArray, double first, double last, int len) {
	double step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { fArray[i] = first + i * step; }
}

// customized function generating values to fulfill sample grids
inline float fValue1(int i, int j) { return float(1 + i + 1 + j); }
//inline float fValue2(int i, int j) { .... }

// test of fcuLinterp2d
void fLinterp2d_test()
{
	float a = 1.0f;
	float b = 500.0f;
	int nSampleSize = 500;
	int nInterpSize = 5000;

	// sample grid on x axis
	float* xGrid = (float*)malloc(nSampleSize * sizeof(float));
	linspacef(xGrid, a, b, nSampleSize);
	PRINT_HEAD_1D(xGrid, 10);
	// sample grid on y axis
	float* yGrid = (float*)malloc(nSampleSize * sizeof(float));
	linspacef(yGrid, a, b, nSampleSize);
	PRINT_HEAD_1D(yGrid, 10);

	// interpolation grid on x axis
	float* xInterpGrid = (float*)malloc(nInterpSize * sizeof(float));
	linspacef(xInterpGrid, a, b, nInterpSize);
	PRINT_HEAD_1D(xInterpGrid, 10);
	// interpolation grid on y axis
	float* yInterpGrid = (float*)malloc(nInterpSize * sizeof(float));
	linspacef(yInterpGrid, a, b, nInterpSize);
	PRINT_HEAD_1D(yInterpGrid, 10);
	
	// value arrays of the sample/interpolation grids
	float* fpSampleValue = (float*)malloc(nSampleSize * nSampleSize * sizeof(float));
	float* fpInterpValue_cpuResult = (float*)malloc(nInterpSize * nInterpSize * sizeof(float));
	float* fpInterpValue_gpuResult = (float*)malloc(nInterpSize * nInterpSize * sizeof(float));
	
	// initialize values of the sample grid 
	for (int i{ 0 }; i < nSampleSize; ++i) 
		for (int j{ 0 }; j < nSampleSize; ++j) 
			fpSampleValue[i * nSampleSize + j] = fValue1(i, j);
	printf("\nInput: Elements of the sample grid (xGrid→, yGrid↓):\n");
	PRINT_HEAD_ROWWISE_2D(fpSampleValue, nSampleSize, nSampleSize, 5, 5);

	// float-api 2d linear interpolation on CPU
	fLinterp2d(fpInterpValue_cpuResult, fpSampleValue,
		xGrid, nSampleSize, yGrid, nSampleSize,
		xInterpGrid, nInterpSize, yInterpGrid, nInterpSize);
	printf("\nfLinterp2d Output: Elements of the interpolation grid (xInterpGrid→, yInterpGrid↓):\n");
	PRINT_HEAD_ROWWISE_2D(fpInterpValue_cpuResult, nInterpSize, nInterpSize, 5, 5);

	// float-api 2d linear interpolation on CPU
	fcuLinterp2d(fpInterpValue_gpuResult, fpSampleValue,
		xGrid, nSampleSize, yGrid, nSampleSize,
		xInterpGrid, nInterpSize, yInterpGrid, nInterpSize);
	printf("\nfcuLinterp2d Output: Elements of the interpolation grid (xInterpGrid→, yInterpGrid↓):\n");
	PRINT_HEAD_ROWWISE_2D(fpInterpValue_gpuResult, nInterpSize, nInterpSize, 5, 5);

	// compare the results of fcuLinterp2d and fLinterp2d
	float maxv = 0.0f;
	float diff = 0.0f;
	for (int i = 0; i < nInterpSize * nInterpSize; ++i) {
		diff = abs(fpInterpValue_gpuResult[i] - fpInterpValue_cpuResult[i]);
		if (diff > maxv)
			maxv = diff;
	}
	printf("\nMax difference between results of fcuLinterp2d and fLinterp2d: %.4f\n", maxv);

	free(xGrid);
	free(yGrid);
	free(xInterpGrid);
	free(yInterpGrid);
	free(fpSampleValue);
	free(fpInterpValue_cpuResult);
	free(fpInterpValue_gpuResult);
}

void fLinterp1d_test() 
{

}

int main() 
{
	fLinterp2d_test();
	
	fLinterp1d_test();

	return 0;
}
