#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


int THREADS;
int BLOCKS;
int NUM_VALS;

int kernel_calls;

/* Define Caliper region names */
const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";


void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

void array_print(float *arr, int length)
{
    int i;
    for (i = 0; i < length; ++i)
    {
        printf("%1.3f ", arr[i]);
    }
    printf("\n");
}

void array_fill(float *arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i)
    {
        arr[i] = random_float();
    }
}

/*Used ChatGPT for this portion*/
// Function to swap two elements in an array
__device__ void swap(float* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

// Function to partition the data array around a pivot
__device__ int partition(float* data, int left, int right, int pivot) {
    while (left <= right) {
        while (data[left] < pivot)
            left++;
        while (data[right] > pivot)
            right--;

        if (left <= right) {
            swap(data, left, right);
            left++;
            right--;
        }
    }
    return left;
}

// Function to select the median of medians as a pivot
__device__ int selectPivot(float* data, int left, int right) {
    int numElements = right - left + 1;
    int numMedians = (numElements + NUM_SAMPLES_PER_BLOCK - 1) / NUM_SAMPLES_PER_BLOCK;
    int* medians = data + left;

    for (int i = 0; i < numMedians; i++) {
        int medianIndex = left + i * NUM_SAMPLES_PER_BLOCK;
        int median = medians[i] = data[medianIndex];
        for (int j = i - 1; j >= 0 && medians[j] > median; j--) {
            medians[j + 1] = medians[j];
            medians[j] = median;
        }
    }

    return medians[numMedians / 2];
}

// Function to sort each block
__global__ void sortBlocks(float* data, int n) {
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int startIndex = blockId * n;
    int endIndex = startIndex + n - 1;

    int pivot = selectPivot(data, startIndex, endIndex);
    int partitionIndex = partition(data, startIndex, endIndex, pivot);

    // Sort the data for this block using quicksort
    if (startIndex < partitionIndex - 1) {
        CALI_MARK_BEGIN(comp_small);
        sortBlocks(data, partitionIndex - startIndex);
        CALI_MARK_END(comp_small);
    }
    if (partitionIndex < endIndex) {
        CALI_MARK_BEGIN(comp_small);
        sortBlocks(data + partitionIndex, endIndex - partitionIndex + 1);
        CALI_MARK_END(comp_small);
    }
}

// Main function to perform Sample Sort 
void sample_sort(float* data, int n) {
    CALI_MARK_BEGIN(comp_large);
    sortBlocks<<<BLOCKS, THREADS>>>(data, n);
    CALI_MARK_END(comp_large);
}

/*End of ChatGPT portion*/

int main(int argc, char *argv[])
{
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    float *values = (float *)malloc(NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN(data_init);
    array_fill(values, NUM_VALS);
    CALI_MARK_END(data_init);

    start = clock();
    CALI_MARK_BEGIN(comp);
    sample_sort(values, NUM_VALS); /* Inplace */
    CALI_MARK_END(comp);
    stop = clock();

    print_elapsed(start, stop);


    adiak::init(NULL);
    adiak::launchdate();                                             // launch date of the job
    adiak::libraries();                                              // Libraries used
    adiak::cmdline();                                                // Command line used to launch the job
    adiak::clustername();                                            // Name of the cluster
    adiak::value("Algorithm", "SampleSort");                         // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                         // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");                               // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float));                   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);                            // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");                             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs);                            // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS);                        // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                          // The number of CUDA blocks
    adiak::value("group_num", 21);                                   // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and AI (ChatGPT)") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}