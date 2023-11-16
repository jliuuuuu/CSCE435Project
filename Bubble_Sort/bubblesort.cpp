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
const char *data_initialization = "data_initialization"
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";

// create populated vector with random values
std::vector<int> random_fill_function(int size) {
    vector<int> vec(size);
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (auto& random_val: vec) {
        random_val = std::rand() % 100;
    }
    return vec;    
}

// verify sorted vector
int verify_sort(vector<int> &vec) {
    int N = vec.size();
    for(int i = 1; i < N; i++) {
        if(vec[i-1] > vec[i]) {
            return 0;
        }
    }
    return 1;
}

// print vector
void print_vector(vector<int> &vec) {
    for(int val: vec) {
        cout << val << " ";
    }
    cout << endl;
}


// regular bubble sort to be used for parallel bubble sort
void bubble_sort(vector<int> &vec) {
    int N = vec.size();
    for(int i = 0; i < n-1; i++) {
        for(int j = 0; j < n-i-1; j++) {
            if(vec[j] > vec[j+1]) {
                std::swap(vec[j], vec[j+1]);
            }
        }
    }
}


int main(int argc, char** argv[]) {

    CALI_MARK_BEGIN(main_function);

    cali::ConfigManager mgr;
    mgr.start();

    // MPI Information
    int taskid;
    int numtasks;
    int input_size = std::stoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if(taskid == 0) {
        // CALI Timers for data initialization
        CALI_MARK_BEGIN(data_initialization);
        vector<int> host_vec = random_fill_function(input_size);
        CALI_MARK_END(data_initialization);
        // CALI Timers for Comm Large
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        MPI_Send(host_vec.data(), input_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    } else if(taskid == 1) {
         // CALI Timers for Comm Small
        vector<int> device_vec(input_size);
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        MPI_Recv(device_vec.data(), input_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(comp_small); 
        CALI_MARK_END(comm);

        // CALI Timers for Comp Large
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        bubble_sort(device_vec);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        print_vector(device_vec);

        if(verify_sort(device_vec)) {
            cout << "Array is correctly sorted" << endl;
        } else {
            cout << "Array is incorrectly sorted" << endl;
        }
    }

    CALI_MARK_BEGIN(comm);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(comm);
    MPI_Finalize();

    CALI_MARK_END(main_function);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Bubble Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", input_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("group_num", 21); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    return 0;
}