#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
const int TM = 4; // 每个 thread 包含 4 个元素的计算
const int TN = 4;
const int BLOCK_DIM_x = 32; // 每个 Block 包含 32 个线程
const int BLOCK_DIM_y = 32;
const int BM = TM * BLOCK_DIM_x; // Block 层面中，每个 Block 包含 128 个元素的计算
const int BN = TN * BLOCK_DIM_y;
const int BK = 8; // TODO
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
void compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    bool tmp = true;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
        if (error > 1e-5)
        {
            tmp = false;
            printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }
    if (tmp)
    {
        printf("GPU output all right\n");
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    // Block  层面块大小为 BM x BN，每个 block  处理 BM x BN 个线程
    // Thread 层面块大小为 TM x TN，每个 thread 处理 TM x TN 个元素
    __shared__ float SA[BM * BK]; // block 对应到 A，B 矩阵上需要参与计算的矩阵
    __shared__ float SB[BK * BN];
    // threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // 当前线程在矩阵 A, B 中的起始 行索引，列索引
    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    // 根据当前目标矩阵大小进行分块，每 K / BK 作为一块进行独立计算
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};

    for (int ph = 0; ph < width; ph++)
    {
        // 横向遍历归属于同一个 Thread 处理的各项元素（ 行 ）
        for (int index_q = 0; index_q < TM; index_q++)
        {
            // 纵向遍历归属于同一个  thread 处理的各项元素（ 列 ）
            for (int index_k = 0; index_k < BK; index_k++)
            {
                if (indA + index_q < M && index_k + ph * BK < K)
                {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = dA[(indA + index_q) * K + index_k + ph * BK];
                }
                else
                {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = 0.0f;
                }
            }
        }
        // 理论上来说取消这个部分并不会影响到计算，
        // 但是从世纪测试上来看，如果将这个同步删除会导致计算所需要的数据超过当前计算机所能提供的，进而导致报错
        __syncthreads();
        for (int index_v = 0; index_v < TN; index_v++)
        {
            for (int index_k = 0; index_k < BK; index_k++)
            {

                if (indB + index_v < N && index_k + ph * BK < K)
                {

                    SB[index_k * BN + threadIdx.y * TN + index_v] = dB[(index_k + ph * BK) * N + indB + index_v];
                }
                else
                {
                    SB[index_k * BN + threadIdx.y * TN + index_v] = 0.0f;
                }
            }
        }

        __syncthreads();
        // 从共享内存 SA，SB 中读取对应元素计算 C=A+B
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[(threadIdx.x * TM + index_q) * BK + index_k] * SB[index_k * BN + threadIdx.y * TN + index_v];
                }
            }
        }
        __syncthreads();
    }
    // 将 tmp 局部变量结果写回 dC 中
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    // Block  层面块大小为 BM x BN，每个 block  处理 BM x BN 个线程
    // Thread 层面块大小为 TM x TN，每个 thread 处理 TM x TN 个元素
    __shared__ float SA[BM * BK]; // block 对应到 A，B 矩阵上需要参与计算的矩阵
    __shared__ float SB[BK * BN];
    // 当前 Block 在 x 方向的起始元素序号
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // 当前线程在线程块中的线性索引（第几个 thread）
    int smem_a_m = tid % 128;                         // 矩阵 A 行索引
    int smem_a_k = tid / 128;                         // 矩阵 A 列索引
    int smem_b_k = tid % 8;                           // 矩阵 B 行索引
    int smem_b_n = tid / 8;                           // 矩阵 B 列索引
    for (int ph = 0; ph < width; ph++)
    {
        // 通过过对索引的计算，每个线程只需要将当前线程所对应的元素加载到计算机
        // 减少了重复读取的情况
        if (indA + smem_a_m < M && smem_a_k + ph * BK < K)
        {
            SA[smem_a_m * BK + smem_a_k] = dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
        }
        else
        {
            SA[smem_a_m * BK + smem_a_k] = 0.0f;
        }
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K)
        {

            SB[smem_b_k * BN + smem_b_n] = dB[(smem_b_k + ph * BK) * N + indB + smem_b_n];
        }
        else
        {
            SB[smem_b_k * BN + smem_b_n] = 0.0f;
        }

        __syncthreads();
        // 与之前相同，只是在索引上变更
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    // 与原先相同，计算整个线程所对应的矩阵, 只是在索引上变更
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixOrigin(float *dA, float *dB, float *dC, int M, int K, int N)
{

    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    float tmp[TM][TN] = {0.0f}; // 线程私有的
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            // 不断读取 A，B 的对应部分算
            if (indA + index_q < M && indB + index_v < N)
            {
                for (int s = 0; s < K; s++)
                {
                    tmp[index_q][index_v] += dA[(indA + index_q) * K + s] * dB[s * N + indB + index_v];
                }
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            // 然后再不断的写回
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q][index_v];
            }
        }
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

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
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
    printf("CPU time:%.2f second\n", ela);
    compare(hostC, serialC, M, N);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
