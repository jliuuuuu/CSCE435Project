#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *main_function = "main_function";
const char *data_initialization = "data_initialization";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *correct_check = "correct_check";
const char *cudaMemoryCopy= "cudaMemoryCopy";

int BLOCKS;
int THREADS;
int NUM_VALS;
int SORT_TYPE;
std::string SORT_TYPE_STRING;



int random_int() {
    return static_cast<int>(rand());
}

void generate_data(int *data, size_t arr_size, int sort_type) {
    if (sort_type == 1) {
        srand(time(NULL));
        for (int i = 0; i < arr_size; ++i) {
            data[i] = random_int();
        }
        SORT_TYPE_STRING = "random";
    } else if (sort_type == 2) {
        for (int i = 0; i < arr_size; ++i) {
            data[i] = arr_size - i;
        }
        SORT_TYPE_STRING = "reverse";
    } else if (sort_type == 3) {
        for (int i = 0; i < arr_size; ++i) {
            data[i] = i;
        }
        SORT_TYPE_STRING = "sorted";
    } else if (sort_type == 4) {
        for (int i = 0; i < arr_size; ++i) {
            data[i] = (i <= static_cast<float>(arr_size) * 0.01) ? random_int() : i;
        }
        SORT_TYPE_STRING = "1% perturbation";
    } else {
        std::cerr << "Invalid sort type.\n";
    }    
}

void print_elapsed(clock_t start, clock_t stop) {
    double elapsed_time = static_cast<double>(stop - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed_time << "s\n";
} 

bool verify_sort(int *data, int arr_size) {
    for(int i = 1; i < arr_size; i++) {
        if(data[i-1] > data[i]) {
            return false;
        }
    }
    return true;
}

__global__ void oswap(int *data, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ((k%2 != 0) && (k < n-2) && (data[k] >= data[k+1])) { // swap data if odd
        int swap = data[k];
        data[k] = data[k+1];
        data[k+1] = swap;
    }
}

__global__ void eswap(int *data, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ((k%2 == 0) && (k < n-1) && (data[k] >= data[k+1])) { // swap data if even
        int swap = data[k];
        data[k] = data[k+1];
        data[k+1] = swap;
    }
}

void bubble_sort(int *vals) {
    int *dim_values;
    size_t local_size = NUM_VALS * sizeof(int);
    cudaMalloc(&dim_values, local_size);


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemoryCopy);
    cudaMemcpy(dim_values, vals, local_size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudaMemoryCopy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int i = 0;
    while(i < NUM_VALS) {
        eswap<<<threads, blocks>>>(dim_values, NUM_VALS);
        oswap<<<threads, blocks>>>(dim_values, NUM_VALS);
        i++;
    }
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemoryCopy);
    cudaMemcpy(vals, dim_values, local_size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cudaMemoryCopy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    cudaFree(dim_values);
}

int main(int argc, char *argv[]) {

    CALI_MARK_BEGIN(main_function);
    THREADS = std::stoi(argv[1]);
    NUM_VALS = std::stoi(argv[2]);
    SORT_TYPE = std::stoi(argv[3]);

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    cali::ConfigManager mgr;
    mgr.start();

    clock_t start_time;
    clock_t stop_time;

    CALI_MARK_BEGIN(data_initialization);
    int *rand_vals = (int*)malloc(NUM_VALS * sizeof(int));
    generate_data(rand_vals, NUM_VALS, SORT_TYPE);
    CALI_MARK_END(data_initialization);

    start_time = clock();
    bubble_sort(rand_vals);
    stop_time = clock();

    print_elapsed(start_time, stop_time);

    CALI_MARK_BEGIN(correct_check);
    if(verify_sort(rand_vals, NUM_VALS)) {
        std::cout << "Array is correctly sorted." << std::endl;
    } else {
        std::cout << "Array is not correctly sorted" << std::endl;
    }
    CALI_MARK_END(correct_check);
    CALI_MARK_END(main_function);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Parallel Bubble Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", NUM_VALS);
    adiak::value("SortType", SORT_TYPE_STRING);
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", 21);
    adiak::value("implementation_source", "Online, AI");

    mgr.stop();
    mgr.flush();
}

