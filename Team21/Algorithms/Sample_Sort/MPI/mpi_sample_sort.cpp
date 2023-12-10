#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>
#include <string.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

static int intcompare(const void *i, const void *j)
{
    if ((*(int *)i) > (*(int *)j))
        return (1);
    if ((*(int *)i) < (*(int *)j))
        return (-1);
    return (0);
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int *data, int left, int right)
{
    int pivot = data[left + (right - left) / 2];
    int i = left;
    int j = right;

    while (i <= j)
    {
        while (data[i] < pivot)
        {
            i++;
        }
        while (data[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            swap(&data[i], &data[j]);
            i++;
            j--;
        }
    }
    return i;
}

void quickSort(int *data, int left, int right)
{
    if (left < right)
    {
        int pivotIndex = partition(data, left, right);
        quickSort(data, left, pivotIndex - 1);
        quickSort(data, pivotIndex, right);
    }
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    int num_vals;

    if (argc > 2)
    {
        num_vals = atoi(argv[2]);
    }

    int generation_type;

    if (argc > 1)
    {
        generation_type = atoi(argv[1]);
    }

    MPI_Status status;

    /* Define Caliper region names */
    const char *data_init = "data_init";
    const char *comm = "comm";
    const char *comm_small = "comm_small";
    const char *comm_large = "comm_large";
    const char *comp = "comp";
    const char *comp_small = "comp_small";
    const char *comp_large = "comp_large";
    const char *correctness_check = "correctness_check";
    int rc;

    // define buffers
    int *local_data;
    int *splitters, *global_splitters;
    int *buckets, *bucketbuffer, *local_bucket;
    int *outputbuffer, *output;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);   // my_rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // num_procs

    if (num_procs < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    int data_size = num_vals / num_procs; // Number of integers each process generates

    local_data = (int *)malloc(data_size * sizeof(int));

    // generate random data each time
    srand(time(NULL) + my_rank);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    char type[20];
    // every process generates its own data
    CALI_MARK_BEGIN(data_init);
    switch (generation_type)
    {
    case 0: // Sorted
        strcpy(type, "Sorted");
        for (int i = 0; i < data_size; ++i)
        {
            local_data[i] = i + my_rank * data_size; // Each process generates sorted data
        }
        break;

    case 1: // Reverse Sorted
        strcpy(type, "ReverseSorted");
        for (int i = data_size; i > 0; --i)
        {
            local_data[data_size - i] = i + my_rank * data_size; // Each process generates reverse sorted data
        }
        break;

    case 2: // Random
        strcpy(type, "Random");
        for (int i = 0; i < data_size; ++i)
        {
            local_data[i] = rand(); // Each process generates random data
        }
        break;

    case 3: // Perturbed
        strcpy(type, "1%perturbed");
        for (int i = 0; i < data_size; ++i)
        {
            local_data[i] = i + my_rank * data_size; // Generate sorted data initially
        }
        // Perturb 1% of the data randomly
        for (int i = 0; i < data_size / 100; ++i)
        {
            int index = rand() % data_size;
            local_data[index] = rand() % 100; // Perturb with random values
        }
        break;
    }
    CALI_MARK_END(data_init);

    // sort the local data
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    quickSort(local_data, 0, data_size - 1);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // get splitters from each process's local data
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    splitters = (int *)malloc(sizeof(int) * (num_procs - 1));
    for (int i = 0; i < (num_procs - 1); ++i)
    {
        splitters[i] = local_data[num_vals / (num_procs * num_procs) * (i + 1)];
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // send splitters to process 0
    global_splitters = (int *)malloc(sizeof(int) * num_procs * (num_procs - 1));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(splitters, num_procs - 1, MPI_INT, global_splitters, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // choose from the global splitters
    if (my_rank == 0)
    {
        // sort the splitters first
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        quickSort(global_splitters, 0, num_procs * (num_procs - 1));
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        for (int i = 0; i < num_procs - 1; ++i)
        {
            splitters[i] = global_splitters[(num_procs - 1) * (i + 1)];
        }
    }

    // send chosen global_splitters from process 0 to all processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    MPI_Bcast(splitters, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // create buckets and buffers to send data to other processes
    buckets = (int *)malloc(sizeof(int) * (num_vals + num_procs));

    int j = 0;
    int k = 1;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (int i = 0; i < data_size; ++i)
    {
        if (j < (num_procs - 1))
        {
            if (local_data[i] < splitters[j])
            {
                buckets[((data_size + 1) * j) + k++] = local_data[i];
            }
            else
            {
                buckets[(data_size + 1) * j] = k - 1;
                k = 1;
                j++;
                i--;
            }
        }
        else
        {
            buckets[((data_size + 1) * j) + k++] = local_data[i];
        }
    }
    buckets[(data_size + 1) * j] = k - 1;
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // send buffers
    bucketbuffer = (int *)malloc(sizeof(int) * (num_vals + num_procs));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Alltoall(buckets, data_size + 1, MPI_INT, bucketbuffer, data_size + 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // rearrange bucketbuffer
    local_bucket = (int *)malloc(sizeof(int) * 2 * num_vals / num_procs);

    int count = 1;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (j = 0; j < num_procs; ++j)
    {
        k = 1;
        for (int i = 0; i < bucketbuffer[(num_vals / num_procs + 1) * j]; i++)
        {
            local_bucket[count++] = bucketbuffer[(num_vals / num_procs + 1) * j + k++];
        }
    }
    local_bucket[0] = count - 1;
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // sort local_bucket
    int num_elements_to_sort = local_bucket[0];
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    qsort((char *)&local_bucket[1], num_elements_to_sort, sizeof(int), intcompare);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // gather the sorted sections at process 0
    if (my_rank == 0)
    {
        outputbuffer = (int *)malloc(sizeof(int) * 2 * num_vals);
        output = (int *)malloc(sizeof(int) * num_vals);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(local_bucket, 2 * data_size, MPI_INT, outputbuffer, 2 * data_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(correctness_check);
    // rearrange output buffer for correctnesscheck
    if (my_rank == 0)
    {
        count = 0;
        for (j = 0; j < num_procs; ++j)
        {
            k = 1;
            for (int i = 0; i < outputbuffer[2 * num_vals / num_procs * j]; ++i)
            {
                output[count++] = outputbuffer[(2 * num_vals / num_procs) * j + k++];
            }
        }

        // // print output
        // printf("Output: ");
        // for (int i = 0; i < num_vals; i++)
        // {
        //     printf("%d ", output[i]);
        // }

        int sorted = 1;

        for(int i = 1; i < num_vals; i++) {
            if(output[i - 1] > output[i]) {
                sorted = 0;
                break;
            }
        }

        if(sorted) {
            printf("sorted");
        }
        else {
            printf("not sorted");
        }

        free(outputbuffer);
        free(output);
    }
    CALI_MARK_END(correctness_check);

    printf("inputType: %s", type);

    adiak::init(NULL);
    adiak::launchdate();                         // launch date of the job
    adiak::libraries();                          // Libraries used
    adiak::cmdline();                            // Command line used to launch the job
    adiak::clustername();                        // Name of the cluster
    adiak::value("Algorithm", "SampleSort");     // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");     // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", num_vals);         // The number of elements in input dataset (1000)
    adiak::value("InputType", type);             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs);        // The number of processors (MPI my_ranks)
    // adiak::value("num_threads", num_threads);                        // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                          // The number of CUDA blocks
    adiak::value("group_num", 21);                                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and AI (ChatGPT)"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    free(local_data);
    free(splitters);
    free(global_splitters);
    free(buckets);
    free(bucketbuffer);
    free(local_bucket);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}