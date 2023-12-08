#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <algorithm>
#include <vector>

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

__device__ void merge(int* array, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        }
        else {
            temp[k++] = array[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = array[i++];
    }

     while (j <= right) {
        temp[k++] = array[j++];
    }

    for (int i = left; i <= right; i++) {
        array[i] = temp[i];
    }

}

__global__ void mergeSort(int* array, int* temp, int size) {
    for (int i = 1; i <= size - 1; i = 2 * i) {
        for (int j = 0; j < size - 1; j += 2 * i) {
            int mid = min(j + i - 1, size - 1);
            int rightEnd = min(j + 2 * i - 1, size - 1);
            merge(array, temp, j, mid, rightEnd);
        }
    }
}

int main(int argc, char **argv) {
    // input processing
    size_t arrSize;
    int inputType;
    int numProcessors;
    if (argc > 3) {
        numProcessors = std::stoi(argv[1]);
        arrSize = std::stoi(argv[2]);
        inputType = std::stoi(argv[3]);
    } else {
        printf("2 inputs needed: <num processors> <array size> <input type> \n");
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


    int array[arrSize];
    int* d_array, *d_temp;

    data_init(arrSize, array, inputType);

    // Allocate memory on the device
    cudaMalloc((void **)&d_array, sizeof(int) * arrSize);
    cudaMemcpy(d_array, array, sizeof(int) * arrSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_temp, sizeof(int) * arrSize);

    // Launch the parallel merge sort
    mergeSort<<<1, 1>>>(d_array, d_temp, arrSize);

    // Copy the sorted array back to the host
    cudaMemcpy(array, d_array, arrSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < arrSize; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");

    bool isSorted = validate_sorted_arr(arrSize, array);
    printf("is array sorted: %s", isSorted ? "true" : "false");

    // Free allocated memory
    cudaFree(d_array);
    cudaFree(d_temp);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();    
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", arrSize);
    adiak::value("InputType", inputTypeStr);
    adiak::value("num_procs", numProcessors);
    adiak::value("group_num", 21);
    adiak::value("implementation_source", "Online, AI");

    return 0;
}
