#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0;
    if (row < M && col < N)
    {
        for (int s = 0; s < K; s++) // foreach K
        {
            tmp += dA[row * K + s] * dB[s * N + col];
        }
        dC[row * N + col] = tmp;
    }
}
template <int BLOCK_DIM> // the number of threads in block
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0.0f;
    // 为当前 BLOCK 可以访问的 A,B 矩阵元素开辟对应的共享内存
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM][BLOCK_DIM];
    int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    for (int ph = 0; ph < width; ph++)
    // 将每 BLOCK_DIM 个部分作为一个块进行分块计算
    // 但是我很好奇这个部分真的会执行超过一次么？
    {
        // 根据需要初始化对应 A，B 矩阵的共享内存
        if (row < M && threadIdx.y + ph * BLOCK_DIM < K)
        {
            SA[threadIdx.x][threadIdx.y] = dA[row * K + threadIdx.y + ph * BLOCK_DIM];
        }
        else
        {
            SA[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if (threadIdx.x + ph * BLOCK_DIM < K && col < N)
        {
            SB[threadIdx.x][threadIdx.y] = dB[(threadIdx.x + ph * BLOCK_DIM) * N + col];
        }
        else
        {
            SB[threadIdx.x][threadIdx.y] = 0.0f;
        }
        // 保证所有共享内存已经写入完成, 此处在不同线程中各进行了一步写入
        // 因此必须要确保所有线程中的写入已经完成才能继续运行，否则会 UB
        __syncthreads();
        // 对于已经位于共享内存中的数据进行 sum 运算
        for (int s = 0; s < BLOCK_DIM; s++)
        {
            tmp += SA[threadIdx.x][s] * SB[s][threadIdx.y];
        }
        __syncthreads();
    }
    // 写回
    if (row < M && col < N)
    {
        dC[row * N + col] = tmp;
    }
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 采用 32 为 BLOCK_DIM 主要是因为一个 wrap 是由 32 个线程组成的
    // the issue of instruciton is base on wrap but not single wrap,
    // thus use this could strink the idle threads.
    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;

    // 确保有足够多的线程块覆盖整个矩阵的行列
    int num_blocks_x = (M + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    printf("block_dim: (%d, %d, %d)", BLOCK_DIM_x, BLOCK_DIM_y, 1);
    printf("grid_dim:  (%d, %d, %d)", num_blocks_x, num_blocks_y, 1);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;

    // NOTE: 预热 GPU 机器
    // matrixKernel1st<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    matrixKernel2nd<32><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        // matrixKernel1st<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        matrixKernel2nd<32><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    hostMatrix(hostA, hostB, hostC, M, K, N);
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    float error = compare(hostC, serialC, M, N);
    printf("CPU time:%.2f second\n", ela);
    printf("The error between CPU and GPU: %.4e\n", error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
