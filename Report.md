# CSCE 435 Group project

## 0. Group number: 21

## 1. Group members:
1. Justin Liu
2. Sam Yang
3. Lucas Ma
4. Tanya Trujillo

---

## 2. _due 10/25_ Project topic
For this project, our group will be implementing sorting algorithms (bubble sort, sample sort and merge sort) in MPI and CUDA. We will examine and compare different algorithm efficencies and times through a variety of different inputs.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

Algorithms:
  - Merge Sort - MPI
  - Merge Sort - CUDA 
  - Bubble Sort - MPI
  - Bubble Sort - CUDA
  - Sample Sort - MPI
  - Sample Sort - CUDA

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

Merge Sort MPI
Pseudocode:
```
void merge(array, left, mid, right):
    i = left, j = mid + 1, k = 0
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }


void mergeSort(array, left, right):
    if (left < right):
        mid = left + (right - left) / 2

        mergeSort(array, left, mid)
        mergeSort(array, mid + 1, right)

        merge(array, left, mid, right)

main:
    set up MPI: MPI_Init, MPI_Comm_size, MPI_Comm_rank
    MPI_Comm_split
    chunk_size = arr_size / world_size;
    MPI_Scatter(data, chunk_size)

    // local sort on each process
    temp = malloc(chunk_size * sizeof(int))
    mergeSort(local_chunk, temp, 0, chunk_size - 1)

    MPI_Gather

    if (world_rank == 0):
        mergeSort(data, 0, arr_size - 1)
        validate_sorted_arr(arrSize, data)
    MPI_Comm_free
    MPI_Finalize
    
```
Citation: https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html#

Merge Sort CUDA

```
_device__ void merge(array, temp, left, mid, right):
    // same as above

__global__ void mergeSort(array, temp, size):
    // iterative merge sort because cannot perform recursive functions inside CUDA

main:
    set up and format data

    cudaMalloc((void **)&array, sizeof(int) * arrSize)
    cudaMemcpy(array, array, sizeof(int) * arrSize, cudaMemcpyHostToDevice)
    cudaMalloc((void **)&temp, sizeof(int) * arrSize)

    mergeSort<<<1, 1>>>(array, temp, arrSize);

    cudaMemcpy(array, arrSize * sizeof(int), cudaMemcpyDeviceToHost);

    validate_sorted_arr(arrSize, array);

```


Bubble Sort
Pseudocode:
```
bubbleSort(array)
Begin
	Sorted = false
	N = len(array)
	While not sorted
		Sorted = true
		For i = 0 to N-1
			Swap array[i] and array[i+1]
			Sorted = false
	End while
End
```
Citation: https://medium.com/@keshav519sharma/parallel-bubble-sort-7ec75891afff

Odd-Even Transposition Sort
```
procedure ODD-EVEN PAR(n)
begin
	id := process’s label
	for i:= 1 to n do
	begin 
		if i is odd then
			if id is odd
				compare-exchange min(id+1);
			else 
				compare-exchange max(id - 1);

		if i is even then
			if id is even then
				compare-exchange min(id + 1);
			else 
				compare-exchange max(id - 1);
	end for
end ODD-EVEN PAR
```
Citation: Design of Parallel Algorithms Slides, Olga Pierce (TAMU)

Sample Sort Pseudocode:
1. Data Initialization
    - For MPI: Have each process create its own local data
    - For CUDA: Create data array at the beginning
2. Create splitters: 
    - MPI: Select splitters from local data and send those splitters to other processes
        - MPI_Allgather() -> to send local splitters to global splitters
    - CUDA: User kernel calls to split up the data
3. Sort Splitters
4. Partition the data:
    - MPI: set up buffers to do MPI_Send and MPI_Receive to deliver the data for each splitter to the other processes
    - CUDA: send data using cudaMemcpy
5. Sort the data:
    - MPI and CUDA: use quicksort to sort the local data on each process
    - CUDA calls quicksort in each kernel
6. Merge the data from each process
     - MPI: MPI_Allgather
     - CUDA: cudaMemcpy
  
Citation: https://en.wikipedia.org/wiki/Samplesort

### 2c. Evaluation plan - what and how will you measure and compare

At the moment, for each of the algorithms, we plan on testing input sizes of arrays in a range of 2^2 to 2^24.
The input types that we will be using to test are ints. 
We test weak scaling and strong scaling for all algorithms.
We test different input sizes as well as different input types(sorted, random, reverse sorted, 1% perturbed).


## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

## 11/08/2023 Update
- We have not finished our project implementation quite yet
- Some of the problems we encountered were time constraints on testing the algorithms
- For this reason, we also do not have any caliper files at this time in our project

## Merge Sort Performance Evaluation


## Bubble Sort Performance Evaluation
### Algorithm Description
Bubble Sort is a sorting algorithm that works by repeatedly swapping adjacent elements if they are in the wrong order.
- Distribute Data: divide list into parts and distribute among processes. Each process will work on its on portion of the list.
- Local Bubble Sort: each process performs a bubble sort on its part
- Exchange & Merge: after bubble sort, exchange elements with neighboring processes and perform comparisons and swaps to ensure that it is sorted. Uses MPI_Barrier

### What are you comparing?
This algorithm will compare different array sizes with different numbers of processors to indicate what the most optimal conditions are for parallelism to have the greatest impact on speedup and performance.

### Problem sizes
- array sizes: 2^16, 2^20, 2^24, 2^28
- number of processors: 2, 4, 8, 16, 32, 64

### Amount of resources
- Grace HPC Cluster (grace.hprc.tamu.edu)

### Figures
We currently do not have any figures for bubble sort at the moment

## Sample Sort Performance Evaluation
### Algorithm Description
Sample Sort is a sorting algorithm similar to the Quick Sort algorithm in which it divides and conquers but uses several pivot elements to sort the array in several segments.
- Select Pivots: Each process selects pivots (splitters) to send to the other processes
- Sort Pivots: The global pivots collected from each process will be sorted
- Distribute Data: Processes send each element to the required segments of other processes
- Local Sorting: Each process sorts its own elements

### What are you comparing?
This algorithm will be comparing weak and strong scaling for different sized problems and different number of threads/processes as well as the speedup 
for different implementations in MPI and CUDA. 

### Problem sizes
- array sizes: 2^16, 2^20, 2^24, 2^28
- number of processors: 2, 4, 8, 16, 32, 64, 128

### Amount of resources
- Grace HPC Cluster (grace.hprc.tamu.edu)
    - 1 CPU
    - 1 GPU
### Figures
There are currently no figures for the sample sort algorithm

## 4. Performance evaluation

Include detailed analysis of computation performance, communication performance. 
Include figures and explanation of your analysis.

### Sample Sort Performance Analysis
  #### MPI
  For the MPI implementation of Sample Sort I was able to run it with 2 processes up to 1024 processes. I was also able to run it at all of the different input sizes that ranged from 2^16 to 2^28. However, some of the sizes such as 2^16 did not want to work for some reason at a certain number of processes and with different input types. I think this might have been due to my code implementation. Furthermore, the highest size 2^28 did not want to work either at the higher number of processes and I noticed that I would often have to increase the memory size of the nodes in the job files. I believe this was due to how I implemented the Sample Sort logic of creating the buckets which I did for every single process with the same size. Thus, this could explain that sometimes, when a process had more values that fell into its bucket, it might not complete because its bucket needed more memory. 

  When I went to do the Jupyter plots using the cali files, I had not realized that my inputType variable had not been reading properly into the adiak caliper readings. Therefore, I decided to use Excel to create the graphs. I was only able to do Weak and Strong scaling plots for the comp_large, comm, and main average times. 

  ##### Strong Scaling comp_large 
  For the comp_large component, for most of the input sizes, the graphs show that as the number of processes increased, the computation decreased which is pretty normal. On some of the graphs, they look all over the place because I was not able to get certain runs to work and that is visible in the first plot alone. 

![Alt text](imagesforreport/../../.vscode/imagesforreport/image-2.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-3.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-4.png)
![Strong Scaling](imagesforreport/../../.vscode/imagesforreport/image.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-6.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-7.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-8.png)
  ##### Strong Scaling comm
  For the comm component, most of the graphs show that as the number of processes increased, the communication also increased and this is normal because there is more processes to communicate between. The graph for input size 2^28 shows communication increase and then decrease and this is because I was not able to get the runs at the last two processes sizes and again, I think this was due to memory issues in the logic of my code. 

![Alt text](imagesforreport/../../.vscode/imagesforreport/image-9.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-11.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-13.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-15.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-17.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-19.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-21.png)
  ##### Strong Scaling main
For the main plots, we can see that a lot of the time was taken at the beginning and at the end for various plots and this was due to the data initialization and the correctness check being the key factors in this. We can also see that for the most part, the time each of the input types took up remained constant throughout the runs, with Sorted being the fastest and 1%perturbed being the slowest.

![Alt text](imagesforreport/../../.vscode/imagesforreport/image-10.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-12.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-14.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-16.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-18.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-20.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-22.png)
##### Weak Scaling comp_large
For the weak scaling comp_large components, the graphs show that the computation decreased with each increase in the number of processors and also with the input size.

![Alt text](imagesforreport/../../.vscode/imagesforreport/image-23.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-24.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-25.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-26.png)
##### Weak Scaling comm
For the comm component, we can see that the communication increased with both the number of processors and the input size. The highest input size 2^28 gave the most outliers in the data. 

![Alt text](imagesforreport/../../.vscode/imagesforreport/image-27.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-28.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-29.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-30.png)
##### Weak Scaling main
For the main component, we can see that the data stayed mostly consistent across each of the different input sizes and remained constant throughout each increase in processors. 
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-31.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-32.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-33.png)
![Alt text](imagesforreport/../../.vscode/imagesforreport/image-34.png)

### CUDA
I was not able to get my CUDA implementation of Sample Sort to work in time. I believe this was due to not being able to properly get the buckets to send and receive data across each of the BLOCKS in the kernel calls in order to properly parallelize the algorithm.

## 5. Presentation
Plots for the presentation should be as follows:
- For each implementation:
    - For each of comp_large, comm, and main:
        - Strong scaling plots for each InputSize with lines for InputType (7 plots - 4 lines each)
        - Strong scaling speedup plot for each InputType (4 plots)
        - Weak scaling plots for each InputType (4 plots)

Analyze these plots and choose a subset to present and explain in your presentation.

## 6. Final Report
Submit a zip named `TeamX.zip` where `X` is your team number. The zip should contain the following files:
- Algorithms: Directory of source code of your algorithms.
- Data: All `.cali` files used to generate the plots seperated by algorithm/implementation.
- Jupyter notebook: The Jupyter notebook(s) used to generate the plots for the report.
- Report.md
