#include <iostream>
#include <vector>
#include <algorithm>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void data_init(size_t arrSize, int *data, int inputType) {
    if (inputType == 0) {
        // random implementation
        for (size_t i = 0; i < arrSize; i++) {
            data[i] = rand() % (arrSize * 10);
        }
    } else if (inputType == 1) {
        // sorted implementation
        for (size_t i = 0; i < arrSize; i++) {
            data[i] = i;
        }
    } else if (inputType == 2) {
        // reverse sorted implementation
        for (size_t i = 0; i < arrSize; i++) {
            data[i] = arrSize - i;
        }
    } else {
        // 1% perturbed
        for (int i = 0; i < arrSize; i++) {
            data[i] = i;
        }
        int randPercentage = arrSize / 100;
        for (int i = 0; i < randPercentage; i++) {
            int index = rand() % arrSize;
            data[index] = rand() % (arrSize * 10);
        }
    }
}

bool validate_sorted_arr(size_t arrSize, int *data) {
    for (size_t i = 1; i < arrSize; i++) {
        if (data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}

void merge(int arr[], int temp[], int left, int mid, int right) {
    int i = left, j = mid + 1, k = 0;

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

void mergeSort(int arr[], int temp[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, temp, left, mid);
        mergeSort(arr, temp, mid + 1, right);

        merge(arr, temp, left, mid, right);
    }
}


int main(int argc, char *argv[]) {

    // input processing
    size_t arrSize;
    int inputType;
    if (argc > 2) {
        arrSize = std::stoi(argv[1]);
        inputType = std::stoi(argv[2]);
    } else {
        printf("2 inputs needed: <array size> <input type> \n");
        printf("input types: 0= random, 1= sorted, 2= reverse sorted, 3= 1%% perturbed");

        return 0;
    }

    std::string inputTypeStr = "";
    if (inputType == 0) {
        inputTypeStr = "Random";
    } else if (inputType == 1) {
        inputTypeStr = "Sorted";
    } else if (inputType == 2) {
        inputTypeStr = "Reverse Sorted";
    } else {
        inputTypeStr = "1% Perturbed";
    }
    
    
    // init MPI
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &new_comm);


    CALI_MARK_BEGIN("main");


    CALI_MARK_BEGIN("data_init");
    int *data = new int[arrSize];
    data_init(arrSize, data, inputType);
    int arr_size = arrSize;
    CALI_MARK_END("data_init");

    // Checking if data init works
    // if (world_rank == 0) {
    //     printf("Unsorted Array: ");
    //     for (int i = 0; i < arr_size; i++) {
    //         printf("%d ", data[i]);
    //     }
    //     printf("\n");
    // }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    int chunk_size = arr_size / world_size;

    // scatter chunks of the array to each process
    int local_chunk[chunk_size];

    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(data, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // local sort on each process
    int* temp = (int*)malloc(chunk_size * sizeof(int));
    mergeSort(local_chunk, temp, 0, chunk_size - 1);


    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_chunk, chunk_size, MPI_INT, data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (world_rank == 0) {
        printf("world_size: %d \n", world_size);
        printf("arr size: %d \n", arrSize);

        int* temp = (int*)malloc(arr_size * sizeof(int));
        mergeSort(data, temp, 0, arr_size - 1);

        end_time = MPI_Wtime();

        // printf("Sorted Array: ");
        // for (int i = 0; i < arr_size; i++) {
        //     printf("%d ", data[i]);
        // }
        // printf("\n");

        printf("Total time taken: %f seconds\n", end_time - start_time);

        CALI_MARK_BEGIN("correctness_check");
        bool isValid = validate_sorted_arr(arrSize, data);
        CALI_MARK_END("correctness_check");
        printf("is array sorted: %s", isValid ? "true" : "false");

        free(temp);
    }

    free(temp);
    CALI_MARK_END("main");

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();    
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", arrSize);
    adiak::value("InputType", inputTypeStr);
    adiak::value("num_procs", world_size);
    adiak::value("group_num", 21);
    adiak::value("implementation_source", "Online, AI");

    // Finalize MPI
    MPI_Comm_free(&new_comm);
    MPI_Finalize();

    return 0;
}






