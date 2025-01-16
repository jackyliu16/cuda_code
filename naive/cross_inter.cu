#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK_CUDA(call)                                                                       \
    do                                                                                         \
    {                                                                                          \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess)                                                                \
        {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

__global__ void crossAddKernel(float *deviceData, int *cnt)
{
    for (int strip = 1; strip < blockDim.x; strip *= 2)
    {
        if (threadIdx.x % (2 * strip) == 0)
        {
            deviceData[threadIdx.x] += deviceData[threadIdx.x + strip];
            atomicAdd(cnt, 1);
        }
        // NOTE: 如果没有这个，可能会出现当前层次没完成时先开始计算下一个层次的情况
        __syncthreads();
    }
}

__global__ void interAddKernel(float *deviceData, int *cnt)
{
    for (int strip = blockDim.x / 2; strip > 0; strip /= 2)
    {
        if (threadIdx.x < strip)
        {
            deviceData[threadIdx.x] += deviceData[threadIdx.x + strip];
            atomicAdd(cnt, 1);
        }
        __syncthreads();
    }
}

int main()
{

    // init
    float *hostData, *deviceData;
    int n = 1024000;
    int *cnt_a, *cnt_b, *da_cnt, *db_cnt;

    hostData = (float *)malloc(n * sizeof(float));
    cnt_a = (int *)malloc(sizeof(int));
    cnt_b = (int *)malloc(sizeof(int));
    *cnt_a = 0;
    *cnt_b = 0;

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        hostData[i] = i;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++)
    {
        sum += i;
    }
    gettimeofday(&end, NULL);
    printf("EXCEPT: %.04f; TIME: %f\n", sum, (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);

    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356
    cudaMalloc((void **)&deviceData, n * sizeof(float));
    cudaMalloc(&da_cnt, sizeof(int));
    cudaMemcpy(deviceData, hostData, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(da_cnt, cnt_a, sizeof(int), cudaMemcpyHostToDevice);

    int BLOCK_DIM = 1024;
    int block_x = n / BLOCK_DIM; // 以之为整数倍打开线程，使至少一个线程运行 1024 个单元的计算。
    dim3 grid_dim(block_x, 1, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);

    // Cacluate 1: cross
    float cross_kernel_time;
    cudaEvent_t start_cross, end_cross;
    cudaEventCreate(&start_cross);
    cudaEventCreate(&end_cross);
    cudaEventRecord(start_cross, 0);

    crossAddKernel<<<grid_dim, block_dim>>>(deviceData, da_cnt);

    cudaEventRecord(end_cross, 0);
    cudaEventSynchronize(end_cross);
    cudaEventElapsedTime(&cross_kernel_time, start_cross, end_cross);

    cudaMemcpy(hostData, deviceData, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt_a, da_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    printf("CNT: %d; RESULT: %.04f;\n", *cnt_a, *hostData);

    for (int i = 0; i < n; i++)
    {
        hostData[i] = i;
        sum += i;
    }
    /// Cacluate 2: interleaving
    cudaMalloc(&db_cnt, sizeof(int));
    cudaMemcpy(deviceData, hostData, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db_cnt, cnt_b, sizeof(int), cudaMemcpyHostToDevice);

    float inter_kernel_time;
    cudaEvent_t start_inter, end_inter;
    cudaEventCreate(&start_inter);
    cudaEventCreate(&end_inter);
    cudaEventRecord(start_inter, 0);

    interAddKernel<<<grid_dim, block_dim>>>(deviceData, db_cnt);

    cudaEventRecord(end_inter, 0);
    cudaEventSynchronize(end_inter);
    cudaEventElapsedTime(&inter_kernel_time, start_inter, end_inter);

    cudaMemcpy(hostData, deviceData, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt_b, db_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    printf("CNT: %d; RESULT: %.04f\n", *cnt_b, *hostData);
    CHECK_CUDA(cudaDeviceSynchronize());

    free(hostData);
    free(cnt_a);
    free(cnt_b);
    printf("n = %d\n cross_time: %.4f\n inter_time: %.4f\n", n, cross_kernel_time, inter_kernel_time);
    return 0;
}
