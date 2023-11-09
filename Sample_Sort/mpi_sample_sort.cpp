#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/* Quick Sort Algorithm obtained from ChatGPT*/
// Function to partition the array and return the pivot index
void swapElements(float* arr, int index1, int index2) {
    float temp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = temp;
}

int partition(float* arr, int low, int high) {
    float pivot = arr[low]; // Choose the first element as the pivot
    int left = low + 1;
    int right = high;

    while (true) {
        while (left <= right && arr[left] <= pivot)
            left++;
        while (left <= right && arr[right] >= pivot)
            right--;

        if (left <= right) {
            swapElements(arr, left, right);
        } else {
            break;
        }
    }

    // Swap the pivot element with the element at the right pointer
    swap(arr, low, right);

    return right;
}

// Quicksort function
void quicksort(float* arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);
        quicksort(arr, low, pivotIndex - 1);
        quicksort(arr, pivotIndex + 1, high);
    }
}

/*********/

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
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

int main(int argc, char *argv[])
{
    //CALI_CXX_MARK_FUNCTION;

    int num_vals;
    if (argc == 2)
    {
        num_vals = atoi(argv[1]);
    }
    else
    {
        printf("\n Please provide the size of the matrix");
        return 0;
    }
    int numranks,    /* number of tasks in partition */
        taskid,      /* a task identifier */
        numworkers,  /* number of worker tasks */
        source,      /* task id of message source */
        dest,        /* task id of message destination */
        mtype,       /* message type */
        rows,        /* rows of matrix A sent to each worker */
        offset,      /* used to determine rows sent to each worker */
        i, j, k, rc; /* misc */

    MPI_Status status;

    /* Define Caliper region names */
    const char *data_init = "data_init";
    const char *comm = "comm";
    const char *comm_small = "comm_small";
    const char *comm_large = "comm_large";
    const char *comp = "comp";
    const char *comp_small = "comp_small";
    const char *comp_large = "comp_large";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    if (numranks < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    // set the size of each array in each process
    int n = num_vals / numranks;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // each process generates it's own set of data
    float *values = (float *)malloc(n * sizeof(float));
    array_fill(values, n);

    // starting and ending indeces for each process
    int local_start_index = taskid * (num_vals / numranks);

    int local_end_index = (taskid + 1) * (num_vals / numranks);

    // choose splitters for each process
    int local_splitters_size = numranks - 1;

    float *local_splitters = (float *)malloc(local_splitters_size * sizeof(float));

    /* Written with help from ChatGPT */
    // choose splitters
    for (int i = 0; i < local_splitters_size; ++i)
    {
        int index = (i + 1) * (n / (local_splitters_size + 1));
        local_splitters[i] = values[index];
    }

    /********/

    // send the splitters to everyone
    int global_splitters_size = numranks * local_splitters_size;

    float *global_splitters = (float *)malloc(global_splitters_size * sizeof(float));

    MPI_Allgather(local_splitters, local_splitters_size, MPI_FLOAT, global_splitters, local_splitters_size, MPI_COMM_WORLD);

    // sort splitters
    quicksort(global_splitters, 0, global_splitters_size - 1);

    int buffer_sizes[numranks][];


    adiak::init(NULL);
    adiak::launchdate();                                             // launch date of the job
    adiak::libraries();                                              // Libraries used
    adiak::cmdline();                                                // Command line used to launch the job
    adiak::clustername();                                            // Name of the cluster
    adiak::value("Algorithm", "SampleSort");                         // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                         // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");                               // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float));                   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);                            // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");                             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs);                            // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads);                        // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks);                          // The number of CUDA blocks
    adiak::value("group_num", 21);                                   // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and AI (ChatGPT)") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}