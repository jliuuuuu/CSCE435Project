#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char *main_function = "main_function";
const char *data_initialization = "data_initialization";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *correct_check = "correct_check";
const char *MPI_gather = "MPI_gather";
const char *MPI_scatter = "MPI_scatter";
const char *MPI_sendrecv_replace = "MPI_sendrecv_replace";

void generate_data(size_t arr_size, int *data, int sort_type) {
    if (sort_type == 1) {
        //random implementation
        for (size_t i = 0; i < arr_size; i++) {
            data[i] = rand() % (arr_size * 10);
        }
    } else if (sort_type == 2) {
        //sorted implementation
        for (size_t i = 0; i < arr_size; i++) {
            data[i] = i;
        }
    } else if (sort_type == 3) {
        //reverse sorted implementation
        for (size_t i = 0; i < arr_size; i++) {
            data[i] = arr_size - i;
        }
    } else {
        // 1% perturbed
        for (int i = 0; i < arr_size; i++) {
            data[i] = i;
        }
        int randPercentage = arr_size / 100;
        for (int i = 0; i < randPercentage; i++) {
            int index = rand() % arr_size;
            data[index] = rand() % (arr_size * 10);
        }
    }
}

// verify sorted vector
bool verify_sort(size_t arr_size, int *data) {
    int N = arr_size;
    for(size_t i= 1; i < N; i++) {
        if(data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}

// regular bubble sort to be used for parallel bubble sort
void bubble_sort(int arr[], int arr_size) {
    for(int i = 0; i < arr_size-1; i++) {
        for(int j = 0; j < arr_size-i-1; j++) {
            if(arr[j] > arr[j+1]) {
                std::swap(arr[j], arr[j+1]);
            }
        }
    }
}

void parallel_bubble_sort(int *data, int local_size, int world_rank, int world_size) {
    int *local_data = new int[local_size];

    CALI_MARK_BEGIN(MPI_scatter);
    MPI_Scatter(data, local_size, MPI_INT, local_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_scatter);

    for (int i = 0; i < world_size; ++i) {
        if (i % 2 == 0) { 
            if (world_rank % 2 == 0 && world_rank < world_size - 1) {
                CALI_MARK_BEGIN(MPI_sendrecv_replace);
                MPI_Sendrecv_replace(local_data, local_size, MPI_INT, world_rank+1, 0, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(MPI_sendrecv_replace);

                CALI_MARK_BEGIN(comp_large);
                bubble_sort(local_data, local_size);
                CALI_MARK_END(comp_large);

            } else if (world_rank % 2 != 0) {
                CALI_MARK_BEGIN(MPI_sendrecv_replace);
                MPI_Sendrecv_replace(local_data, local_size, MPI_INT, world_rank-1, 0, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(MPI_sendrecv_replace);

                CALI_MARK_BEGIN(comp_large);
                bubble_sort(local_data, local_size);
                CALI_MARK_END(comp_large);
            }
        } else { 
            if (world_rank % 2 != 0 && world_rank < world_size - 1) {
                CALI_MARK_BEGIN(MPI_sendrecv_replace);
                MPI_Sendrecv_replace(local_data, local_size, MPI_INT, world_rank+1, 0, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(MPI_sendrecv_replace);

                CALI_MARK_BEGIN(comp_large);
                bubble_sort(local_data, local_size);
                CALI_MARK_END(comp_large);

            } else if (world_rank % 2 == 0 && world_rank > 0) {
                CALI_MARK_BEGIN(MPI_sendrecv_replace);
                MPI_Sendrecv_replace(local_data, local_size, MPI_INT, world_rank-1, 0, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(MPI_sendrecv_replace);

                CALI_MARK_BEGIN(comp_large);
                bubble_sort(local_data, local_size);
                CALI_MARK_END(comp_large);
            }
        }
    }

    CALI_MARK_BEGIN(MPI_gather);
    MPI_Gather(local_data, local_size, MPI_INT, data, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_gather);
    delete[] local_data;

}

int main(int argc, char* argv[]) {

    size_t arr_size;
    int sort_type;

    CALI_MARK_BEGIN(main_function);
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_size, &new_comm);

    if (argc > 2) {
        arr_size = std::stoi(argv[1]);
        sort_type = std::stoi(argv[2]);
    } else {
        if (world_rank == 0) {
            std::cout << "2 inputs needed: <array size> <sort type>\n";
            std::cout << "Sort types: 0= random, 1= sorted, 2= reverse sorted, 3= 1% perturbed\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string sort_type_str;
    if (sort_type == 1) {
        sort_type_str = "random";
    } else if (sort_type == 2) {
        sort_type_str = "sorted";
    } else if (sort_type == 3) {
        sort_type_str = "reverse";
    } else {
        sort_type_str = "1% perturbed";
    }

    CALI_MARK_BEGIN(data_initialization);
    int *data = new int[arr_size];
    generate_data(arr_size, data, sort_type);
    CALI_MARK_END(data_initialization);

    if (world_rank == 0) {
        std::cout << "Unsorted Array: ";
        for (size_t i = 0; i < arr_size; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    int local_size = arr_size / world_size;
    parallel_bubble_sort(data, local_size, world_rank, world_size);

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if (world_rank == 0) {
        std::cout << "Sorted Array: ";
        for (size_t i = 0; i < arr_size; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;

        CALI_MARK_BEGIN(correct_check);
        bool is_sorted = verify_sort(arr_size, data);
        CALI_MARK_END(correct_check);

        std::cout << "Is array sorted: " << (is_sorted ? "true" : "false") << std::endl;
    }

    CALI_MARK_END(main_function);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Parallel Bubble Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", arr_size);
    adiak::value("SortType", sort_type_str);
    adiak::value("num_procs", world_size);
    adiak::value("group_num", 21);
    adiak::value("implementation_source", "Online, AI");

    MPI_Finalize();

    delete[] data;

    return 0;

}

  