#include "CacheLineSize.h"
#include "MatMulExploration.h"

#define N 1024
#define N_TEX 1024
float A[N][N];
float B[N][N];
float C[N][N];

double PCFreq = 0.0;
__int64 CounterStart = 0;

int size;
int hwidth;
int hheight;
int ripplemap[N_TEX];
int data;
int pixels[N_TEX*N_TEX];
static int disturbsize = 64;
static int riprad = 5;

int oldind;
int newind;
int mapind;
int width;
int height;

// A 16byte = 128bit vector struct
struct Vector4 {
	float x, y, z, w;
} typedef Vector4;

//HighRes Performance counter
void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li)) {
		printf("QueryPerformanceFrequency failed!\n");
		exit(-1);
	}
	PCFreq = (double)li.QuadPart / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

//HighRes Performance counter
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return (double) (li.QuadPart - CounterStart) / PCFreq;
}


void initmatrix(void)
{
	register int i=0, j=0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A[i][j] = B[i][j] = (float)i+j;
			C[i][j] = 0.0;
		}
	}
}

/*
Dumb simple & easy matrix multiplication
*/
void matmul(void)
{
	int i=0, j=0, l=0;
	float sum = 0.0;
	for (i = 0; i<N; i++) {
		for (j = 0; j<N; j++) {
			sum = C[j][i];
			for (l = 0; l<N; l++) {
				sum += A[l][i] * B[j][l];
			}
			C[j][i] = sum;
		}
	}
}

/*
Use register for commonly used variable to increase performance
*/
void improved_matmul(void)
{
	register int i=0, j=0, l=0;
	register float sum = 0.0;
	for (i = 0; i<N; i++) {
		for (j = 0; j<N; j++) {
			sum = C[j][i];
			for (l = 0; l<N; l++) {
				sum += A[l][i] * B[j][l]; //use sum so that a register will be used in the assembly and not the cache
			}
			C[j][i] = sum;
		}
	}
}

/*
Break down the matrix into pieces of size BS that can fit inside the cache
*/
void cacheaware_matmul(int BS)
{
	register int i = 0, j = 0, l = 0;
	register int ii = 0, jj = 0, ll = 0;
	register float sum = 0.0;
	for (i = 0; i<N; i += BS) {
		for (j = 0; j<N; j += BS) {
			for (l = 0; l<N; l += BS) {
				for (ii = i; ii<i+BS; ii++) {
					for (jj = j; jj<j+BS; jj++) {
						sum = C[jj][ii];
						for (ll = l; ll<l+BS; ll++) {
							sum += A[ll][ii] * B[jj][ll];
						}
						C[jj][ii] = sum;
					}
				}
			}
		}
	}
}

/*
Break down the matrix into pieces of size BS that can fit inside the cache + use SIMD instructions to accelerate vector multiplication
*/
void custom_asm_matmul(int BS)
{
	register int i = 0, j = 0, l = 0;
	register int ii = 0, jj = 0, ll = 0;
	register float sum = 0.0;
	for (i = 0; i<N; i += BS) {
		for (j = 0; j<N; j += BS) {
			for (l = 0; l<N; l += BS) {
				for (ii = i; ii<i + BS; ii+=1) {
					for (jj = j; jj<j + BS; jj+=1) {
						register __m128 S = _mm_set1_ps(C[jj][ii]);
						for (ll = l; ll<l + BS; ll+=4) {
							register __m128 vA = _mm_setr_ps(A[ll][ii], A[ll + 1][ii], A[ll + 2][ii], A[ll + 3][ii]); //pack 4 elements into vector vA
							register __m128 vB = _mm_setr_ps(B[jj][ll], B[jj][ll + 1], B[jj][ll + 2], B[jj][ll + 3]); //pack 4 elements into vector vB
							__asm
							{
								MOVUPS XMM0, vA //load vA
								MULPS XMM0, vB //multiply vA with VB
								HADDPS XMM0, XMM0 //horizontally add XMM0 to itself so that all elements have the same value
								HADDPS XMM0, XMM0 //requires to adds
								ADDPS XMM0, S //add last value of sum
								MOVSS S, XMM0      // save the return vector
							}
						}
						_mm_store_ss(&C[jj][ii], S); //store S to the matrix
					}
				}
			}
		}
	}
}

//Measures the performance of the 4 different tests of matrix multiplications
int main() {
	
	int BS = sqrtf(CacheSize() / sizeof(float))- ((CacheLineSize() / sizeof(float)));
	while (N % BS != 0) {
		BS -= 1;
	}
	printf("Cache Size = %d bytes\nCache Line = %d bytes\nBlock Size = %d\n", CacheSize(), CacheLineSize(), BS);

	printf("Matrices NxN where size N = %d\n", N);

	StartCounter();
	initmatrix();
	printf("Matrices initialized in %5.3f ms\n", GetCounter());

	StartCounter();
	matmul();
	printf("BasicMatMul finished in %5.3f ms\n", GetCounter());
	initmatrix();

	StartCounter();
	improved_matmul();
	printf("WithRegisterMatMul finished in %5.3f ms\n", GetCounter());
	initmatrix();

	StartCounter();
	cacheaware_matmul(BS);
	printf("CacheAwareMatMul finished in %5.3f ms\n", GetCounter());
	initmatrix();

	StartCounter();
	custom_asm_matmul(BS);
	printf("CacheAwareSIMDAssembly finished in %5.3f ms\n", GetCounter());

	printf("Demo ended\n");

	printf("Press any key to exit...");

	getchar();
	exit(0);
}
