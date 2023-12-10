#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

int kernel_calls;

const char *cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char *cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

/* Define Caliper region names */
const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";

__global__ void chooseSplitters(int* data, int* splitters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (THREADS - 1)) {
        splitters[tid] = data[((tid + 1) * BLOCKS)];
    }
}

// Function to sort each block using paralle bubble sort
__global__ void sortBlocks(int *data, int *splitters)
{
    int tid = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int blockId = blockIdx.x;

    __shared__ int local_data[THREADS];

    local_data[tid] = data[i];
    __syncthreads();

    int j;
    for (j = 0; j < (THREADS / 2); j++)
    {
        if (tid % 2 == 0 && tid < (THREADS - 1))
        {
            if (local_data[tid] > local_data[tid + 1])
            {
                int temp = local_data[tid];
                local_data[tid] = local_data[tid + 1];
                local_data[tid + 1] = temp;
            }
        }
        __syncthreads();

        if (tid % 2 == 1 && tid < (THREADS - 1))
        {
            if (local_data[tid] > local_data[tid + 1])
            {
                int temp = local_data[tid];
                local_data[tid] = local_data[tid + 1];
                local_data[tid + 1] = temp;
            }
        }
        __syncthreads();
    }
    data[i] = local_data[tid];

    for (int j = tid; j < BLOCKS - 1; j += THREADS)
    {
        int index = NUM_VALS / (BLOCKS * BLOCKS) * (i + 1);
        splitters[blockId * (BLOCKS - 1) + j] = data[index]; // Store the splitter
    }
}

__global__ void bucket_histogram(int *data, int *histogram, int *offsets)
{
    __shared__ int local_histogram[BLOCKS];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < BLOCKS)
    {
        local_histogram[tid] = 0;
    }
    __syncthreads();

    int value = data[i];
    int local_offset = atomicAdd(&local_histogram[value], 1);

    __syncthreads();

    if (tid < BLOCKS)
    {
        int count = local_histogram[tid];
        if (count > 0)
        {
            group_offset = atomicAdd(&histogram[tid], count);
            local_histogram[tid] = group_offset;
        }
    }

    __syncthreads();

    offsets[i] = local_offset + local_histogram[value];
}

__global__ void prefix_sum(int *histogram, int *prefix_sum, int n)
{
    __shared__ int temp[];
    tid = threadIdx.x;
    int offset = 1;

    temp[2 * tid] = histogram[2 * tid];
    temp[2 * tid + 1] = histogram[2 * tid + 1];

    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (tid == 0)
    {
        temp[n - 1] = 0;
    }
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int t = temp[ai]

                temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    prefix_sum[2 * tid] = temp[2 * tid];
    prefix_sum[2 * tid + 1] = temp[2 * tid + 1];
}

__global__ bucket_sort_scatter(int *data, int *prefix_sum, int *offsets, int *output)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int value = data[i];
    int scatter_target = offsets[i] + prefix_sum[value];
    output[scatter_target] = value;
}

// Main function to perform Sample Sort
void sample_sort(int *data)
{
    int *dev_values;
    size_t size = NUM_VALS * sizeof(int);

    cudaMalloc((void **)&dev_values, size);

    // select splitters
    // Allocate memory for splitters on the host
    int *host_splitters = (int *)malloc((BLOCKS - 1) * sizeof(int) * BLOCKS);

    // Allocate memory for splitters on the device
    int *dev_splitters;
    cudaMalloc((void **)&dev_splitters, (BLOCKS - 1) * sizeof(int) * BLOCKS);

    // memcpy
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_values, data, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);   /* Number of blocks   */
    dim3 threads(THREADS, 1); /* Number of threads  */

    // sort initial blocks of data and get splitters
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    sortBlocks<<<blocks, threads>>>(dev_values, dev_splitters);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // cuda device synchronize
    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(data, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    // Copy the generated splitters from device to host

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(host_splitters, dev_splitters, (BLOCKS - 1) * sizeof(int) , cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // choose global splitters
    int* global_splitters = (int* malloc(sizeof(int) * (BLOCKS - 1));
    for (int i = 0; i < BLOCKS - 1; ++i)
        {
        global_splitters[i] = host_splitters[(BLOCKS - 1) * (i + 1)];
    }

    int *dev_global_splitters;
    cudaMalloc((void **)&dev_global_splitters, (BLOCKS - 1) * sizeof(int));
    cudaMemcpy(dev_splitters, global_spliters, (BLOCKS - 1) * sizeof(int), cudaMemcpyHostToDevice);

    //cudaDeviceSynchronize();
    int* buckets = (int* malloc(sizeof(int) * (BLOCKS + NUM_VALS));




    cudaFree(dev_values);
}

/*End of ChatGPT portion*/

int main(int argc, char *argv[])
{
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    int generation_type = 2;

    if (argc > 1)
    {
        generation_type = atoi(argv[3]);
    }

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    int *data = (int *)malloc(NUM_VALS * sizeof(int));

    // generate random data each time
    srand(time(NULL));

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    std::string inputType;
    // every process generates its own data
    CALI_MARK_BEGIN(data_init);
    switch (generation_type)
    {
    case 0: // Sorted
        inputType = "Sorted";
        for (int i = 0; i < NUM_VALS; ++i)
        {
            data[i] = i; // Each process generates sorted data
        }
        break;

    case 1: // Reverse Sorted
        inputType = "ReverseSorted";
        for (int i = NUM_VALS; i > 0; --i)
        {
            data[NUM_VALS - i] = i; // Each process generates reverse sorted data
        }
        break;

    case 2: // Random
        inputType = "Random";
        for (int i = 0; i < NUM_VALS; ++i)
        {
            data[i] = rand() % NUM_VALS; // Each process generates random data
        }
        break;

    case 3: // Perturbed
        inputType = "1%perturbed";
        for (int i = 0; i < NUM_VALS; ++i)
        {
            data[i] = i; // Generate sorted data initially
        }
        // Perturb 1% of the data randomly
        for (int i = 0; i < NUM_VALS / 100; ++i)
        {
            int index = rand() % NUM_VALS;
            data[index] = rand() % NUM_VALS; // Perturb with random values
        }
        break;
    default:
        break;
    }
    CALI_MARK_END(data_init);

    // printf("Input Type: %s\n", type);

    clock_t start, stop;

    start = clock();
    CALI_MARK_BEGIN(comp);
    sample_sort(data); /* Inplace */
    CALI_MARK_END(comp);
    stop = clock();

    print_elapsed(start, stop);

    // check correctness
    int sorted = 1;

    for (int i = 1; i < NUM_VALS; i++)
    {
        if (data[i - 1] > data[i])
        {
            sorted = 0;
            break;
        }
    }

    if (sorted)
    {
        printf("Sorted\n");
    }
    else
    {
        printf("Not sorted\n");
    }

    // array_print(data, NUM_VALS);

    adiak::init(NULL);
    adiak::launchdate();                         // launch date of the job
    adiak::libraries();                          // Libraries used
    adiak::cmdline();                            // Command line used to launch the job
    adiak::clustername();                        // Name of the cluster
    adiak::value("Algorithm", "SampleSort");     // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");    // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");             // The datatype of input elements (e.g., double, int, int)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);         // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);  // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", num_procs);                            // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS);                             // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                               // The number of CUDA blocks
    adiak::value("group_num", 21);                                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and AI (ChatGPT)"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}