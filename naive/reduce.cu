#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <cub/block/block_reduce.cuh>
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
float addCpu(float *hostData, int n)
{
    float tmp = 0.0f; // 表示C++中的负无穷
    for (int i = 0; i < n; i++)
    {
        tmp += hostData[i];
    }
    return tmp;
}
template <int BLOCK_DIM> // 类似于 rust 中的 T，作为编译时常量引入
__global__ void addKernel(float *deviceData, int n, float *globalMax, int strategy)
{
    __shared__ float tmpSum[BLOCK_DIM];
    float tmp = 0.0f;
    // 对于当前运行的不同线程而言，在各自线程之中计算 基于所在 threadIdx 偏移的 block 的加总
    for (int id = threadIdx.x; id < n; id += BLOCK_DIM)
    {
        tmp += deviceData[id];
    }
    // 将这个 共享内存 中的所有元素加总，就是我们最终需要计算出的结果
    tmpSum[threadIdx.x] = tmp;
    __syncthreads();
    // 针对已经计算好的 原始数组的分段加总，根据不同的规约进行计算
    if (strategy == 0) // 交叉规约
    {
        for (int step = 1; step < BLOCK_DIM; step *= 2)
        {
            if (threadIdx.x % (2 * step) == 0)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) // 指定由哪个 thread 写回对应的内存
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 1) // 交错规约
    {
        for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
        {
            if (threadIdx.x < step)
            {
                tmpSum[threadIdx.x] += tmpSum[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            globalMax[0] = tmpSum[0];
        }
    }
    else if (strategy == 2) // shuffle wrap
    {
        __shared__ float val[32];
        float data = tmpSum[threadIdx.x];
        data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
        data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
        if (threadIdx.x % 32 == 0)
        {
            val[threadIdx.x / 32] = data;
        }
        __syncthreads();
        if (threadIdx.x < 32)
        {
            data = val[threadIdx.x];
            data += __shfl_down_sync(0xffffffff, data, 16); // 0 + 16, 1 + 17,..., 15 + 31
            data += __shfl_down_sync(0xffffffff, data, 8);  // 0 + 8, 1 + 9,..., 7 + 15
            data += __shfl_down_sync(0xffffffff, data, 4);
            data += __shfl_down_sync(0xffffffff, data, 2);
            data += __shfl_down_sync(0xffffffff, data, 1);
        }

        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            globalMax[0] = data;
        }
    }
    else // BlockReduce
    {
        typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce; //<float,..>里面的float表示返回值的类型
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float block_sum = BlockReduce(temp_storage).Reduce(tmpSum[threadIdx.x], cub::Sum());
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            globalMax[0] = block_sum;
        }
    }
}
int main()
{
    float *hostData;
    int n = 102400;
    int strategy = 0;
    int repeat = 100;
    hostData = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        hostData[i] = (i % 10) * 1e-1; // 生成一系列 0~9 之间的数值
    }
    float hostMax;
    double st, ela;
    st = get_walltime();

    float *deviceData, *globalMax; // globalMax => deviceMax ?
    cudaMalloc((void **)&deviceData, n * sizeof(float));
    cudaMalloc((void **)&globalMax, sizeof(float));
    cudaMemcpy(deviceData, hostData, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    // 每个 grid 有 n/1024 个 block, 每个 block 有 1024 个 threads
    for (int i = 0; i < repeat; i++)
    {
        addKernel<1024><<<grid_dim, block_dim>>>(deviceData, n, globalMax, strategy);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(&hostMax, globalMax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceData);
    cudaFree(globalMax);
    ela = 1000 * (get_walltime() - st);
    printf("n = %d: strategy:%d, GPU use time:%.4f ms, kernel time:%.4f ms\n", n, strategy, ela, ker_time / repeat);
    printf("CPU sum:%.2f, GPU sum:%.2f\n", addCpu(hostData, n), hostMax);
    free(hostData);

    return 0;
}
