#include "cuLinterp.cuh"

// return an evenly spaced 1-d grid of doubles.
std::vector<float> linspaceV(float first, float last, int len) {
	std::vector<float> result(len);
	double step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { result[i] = first + i * step; }
	return result;
}

inline float fValue1(int i, int j) { return float(1 + i + 1 + j); }

void interp2dGpuTest()
{
	float a = 1.0f;
	float b = 500.0f;
	int nSampleSize = 500;
	int nInterpSize = 5000;

	auto xGrid = linspaceV(a, b, nSampleSize);
	printArray1dHead(xGrid.data(), 10);

	auto yGrid = linspaceV(a, b, nSampleSize);
	printArray1dHead(yGrid.data(), 10);

	auto xInterpGrid = linspaceV(a, b, nInterpSize);
	printArray1dHead(xInterpGrid.data(), 10);

	auto yInterpGrid = linspaceV(a, b, nInterpSize);
	printArray1dHead(yInterpGrid.data(), 10);

	float* fpSampleValue = new float[nSampleSize * nSampleSize]{ 0.0f };
	float* fpInterpValue = new float[nInterpSize * nInterpSize]{ 0.0f };

	for (int i{ 0 }; i < nSampleSize; ++i) {
		for (int j{ 0 }; j < nSampleSize; ++j) {
			fpSampleValue[i * nSampleSize + j] = fValue1(i, j);
		}
	}

	std::cout << "Sample value: \n";
	for (int i{ 0 }; i < 5; ++i) {
		for (int j{ 0 }; j < 5; ++j) {
			std::cout << fpSampleValue[i * nSampleSize + j] << " ";
		}
		std::cout << "...\n";
	}
	std::cout << "...\n";

	interp2d_gpu(fpInterpValue, fpSampleValue, xGrid, yGrid, xInterpGrid, yInterpGrid);

	std::cout << "Interpolation value: \n";
	for (int i{ 0 }; i < 5; ++i) {
		for (int j{ 0 }; j < 5; ++j) {
			std::cout << fpInterpValue[i * nInterpSize + j] << " ";
		}
		std::cout << "...\n";
	}
	std::cout << "...\n";

	delete[] fpSampleValue;
	delete[] fpInterpValue;
}


int main() 
{
	interp2dGpuTest();

	return 0;
}
