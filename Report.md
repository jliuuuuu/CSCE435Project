# CSCE 435 Group project

## 0. Group number: 21

## 1. Group members:
1. Justin Liu
2. Sam Yang
3. Lucas Ma
4. Tanya Trujillo

---

## 2. _due 10/25_ Project topic
For this project, our group will be implementing sorting algorithms.

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

Merge Sort
Pseudocode:
```
procedureparallelmergesort(id, n, data, newdata)
Begin
data = sequentialmergesort(data)
for dim = 1 to n
data = parallelmerge(id, dim, data)
endfor
newdata = data
end
```
Citation: https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7 


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
The input types that we will be using to test are floats and ints. 
We plan on testing weak scaling for now.

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
### Algorithm Description
Merge sort is a popular dive-and-conquer algorithm that is often performed with recursion. The steps to the algorithm are:
- Divide: Divide unsorted list into two equal halves. This step is repeated recursively and multiple processors are assigned to divide the list concurrently.
- Conquer: Multiple processors independtly sort their own sublists. This involves furthing dividing and sorting unti lthe base case is reached
- Merge: Merging involves coordinating the merging of sorted sublists into a larger sorted sublist

### What are you comparing?
This algorithm will compare different array sizes with different numbers of processors to indicate what the most optimal conditions are for parallelism to have the greatest impact on speedup and performance.

### Problem sizes
- array sizes: 2^16, 2^20, 2^24, 2^28
- number of processors: 2, 4, 8, 16, 32, 64

### Amount of resources
- Grace HPC Cluster (grace.hprc.tamu.edu)

### Figures
We currently do not have any figures for merge sort at the moment

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

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
