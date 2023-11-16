#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void merge(int arr[], int temp[], int left, int mid, int right) {
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

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

    // char buf[] = "32";
    // int test;
    // test = atoi(buf);

    
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    double start_time, end_time;

    // Get the total number of processes and the rank of the current process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("world_size: %d", world_size);

    // Define the array to be sorted
    int arr[] = {13, 4, 2, 21, 19, 11, 1, 8, 5, 32, 10, 3};
    int arr_size = sizeof(arr) / sizeof(arr[0]);

    // Calculate the size of each chunk for each process
    int chunk_size = arr_size / world_size;

    // Scatter the chunks of the array to each process
    int local_chunk[chunk_size];
    MPI_Scatter(arr, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Perform local sort on each process
    int* temp = (int*)malloc(chunk_size * sizeof(int));
    mergeSort(local_chunk, temp, 0, chunk_size - 1);

    // Gather the sorted chunks back to the root process
    MPI_Gather(local_chunk, chunk_size, MPI_INT, arr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int* temp = (int*)malloc(arr_size * sizeof(int));
        mergeSort(arr, temp, 0, arr_size - 1);

        // Print the sorted array
        end_time = MPI_Wtime();
        printf("Sorted Array: ");
        for (int i = 0; i < arr_size; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        printf("Total time taken: %f seconds\n", end_time - start_time);

        free(temp);
    }

    free(temp);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}






