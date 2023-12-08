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
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *correct_check = "correct_check";
const char *MPI_send = "MPI_Send";
const char *MPI_barrier = "MPI_Barrier";
const char *MPI_recv = "MPI_Recv";

string sort_type_string = "";

int random_int() {
    return std::rand();
}

vector<int> generate_data(size_t size, int sort_type) {
    vector<int> vec(size);
    srand(static_cast<unsigned>(time(nullptr)));
    if(sort_type == 1) {
        for(auto &val: vec) {
            val = random_int();
        }
        sort_type_string = "Random";
    } else if(sort_type == 2) {
        int num_elements = 0;
        for(auto &val: vec) {
            val = num_elements;
            num_elements++;
        }
        sort_type_string = "Sorted";      
    } else if(sort_type == 3) {
        int num_elements = size;
        for(auto &val: vec) {
            val = num_elements;
            num_elements--;
        }
        sort_type_string = "ReverseSorted";        
    } else if(sort_type == 4) {
        for(size_t i = 0; i < size; i++) {
            if(i <= static_cast<float>(size) * 0.01) {
                vec[i] = random_int();
            } else {
                vec[i] =i;
            }
        }
        sort_type_string = "1%perturbed";
    } else {
        printf("Invalid Sorting Type.\n");
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
    for(int i = 0; i < N-1; i++) {
        for(int j = 0; j < N-i-1; j++) {
            if(vec[j] > vec[j+1]) {
                std::swap(vec[j], vec[j+1]);
            }
        }
    }
}


int main(int argc, char* argv[]) {

    CALI_MARK_BEGIN(main_function);

    // MPI Information
    int taskid;
    int numtasks;
    int input_size = std::stoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    cali::ConfigManager mgr;
    mgr.start();


    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if(taskid == 0) {
        // CALI Timers for data initialization
        CALI_MARK_BEGIN(data_initialization);
        vector<int> host_vec = generate_data(input_size, std::stoi(argv[2]));
        CALI_MARK_END(data_initialization);
        // CALI Timers for Comm Large
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        // MPI Send
        CALI_MARK_BEGIN(MPI_send);
        MPI_Send(host_vec.data(), input_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
        CALI_MARK_END(MPI_send);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    } else if(taskid == 1) {
         // CALI Timers for Comm Small
        vector<int> device_vec(input_size);
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        // MPI RECV
        CALI_MARK_BEGIN(MPI_recv);
        MPI_Recv(device_vec.data(), input_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(MPI_recv);
        CALI_MARK_END(comm_small); 
        CALI_MARK_END(comm);

        // CALI Timers for Comp Large
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        bubble_sort(device_vec);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(correct_check);
        if(verify_sort(device_vec)) {
            cout << "Array is correctly sorted" << endl;
        } else {
            cout << "Array is incorrectly sorted" << endl;
        }
        CALI_MARK_END(correct_check);
    }

    CALI_MARK_BEGIN(comm);
    // MPI Barrier
    CALI_MARK_BEGIN(MPI_barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_barrier);
    CALI_MARK_END(comm);

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
    adiak::value("InputType", sort_type_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("group_num", 21); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}
