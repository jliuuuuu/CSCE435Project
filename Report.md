# CSCE 435 Group project

## 1. Group members:
1. Justin Liu
2. Sam Yang 
3. Lucas Ma
4. Tanya Trujillo

---

## 2. _due 10/25_ Project topic
For this project, our group will be implementing sorting algorithms.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For example:
- Merge Sort - MPI on each core
- Merge Sort - MPI + CUDA 
- Bubble Sort - Master/Worker
- Bubble Sort - Single Program Multiple Data
- Odd-Even Transposition Sort - MPI on each core
- Odd-Even Transposition Sort - MPI + CUDA

Merge Sort
Pseudocode:
procedureparallelmergesort(id, n, data, newdata)
Begin
data = sequentialmergesort(data)
for dim = 1 to n
data = parallelmerge(id, dim, data)
endfor
newdata = data
end
Citation: https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7 

Bubble Sort
Pseudocode:
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

Citation: https://medium.com/@keshav519sharma/parallel-bubble-sort-7ec75891afff

Odd-Even Transposition Sort
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
Citation: Design of Parallel Algorithms Slides, Olga Pierce (TAMU)

