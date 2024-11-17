# 15618-F24-Final
Final project for CMU 15-618, F24

Title: Airplane flight routing in parallel

Team members: Jesse Liu (jzliu@andrew.cmu.edu), Enxu Han (enxuh@andrew.cmu.edu)

URL: https://github.com/jliu64/15618-F24-Final

Summary: We aim to optimize the current flight routing algorithm by incorporating parallelism to enhance both performance and efficiency.

Background: 
Flight routing algorithms play a crucial role in optimizing aircraft paths, balancing constraints such as safety, time, and fuel efficiency. With the rise in air traffic and increasingly complex scenarios, traditional algorithms often struggle to efficiently manage multiple obstacles and dynamic constraints in real-time. Approaches like artificial potential fields, Voronoi diagrams, and rapidly-exploring random trees (RRT) provide straightforward solutions but tend to face challenges with scalability and computational efficiency in high-complexity environments. Heuristic and metaheuristic methods offer flexibility and adaptability; however, they are generally computationally demanding and lack determinism in their results.

Efforts to parallelize flight routing algorithms have largely focused on GPU-based solutions, leveraging the high degree of parallelism GPUs provide. However, these approaches often lack adaptability to diverse hardware setups and fail to deliver comprehensive comparisons across frameworks. This has left open questions about their effectiveness across configurations involving varying CPU, memory, and GPU capabilities.

We propose a parallelized implementation of traditional flight routing algorithms using SIMD, MPI, OpenMP, and CUDA. Our objective is to evaluate these approaches under different hardware configurations to determine the optimal solution. This work aims to address the inefficiencies of traditional algorithms in complex scenarios and provide a flexible, scalable framework for real-world flight routing applications.

The challenge:
Workload 
Flight routing involves processing a large, irregular graph with iterative refinements to shortest path estimates. Dependencies arise as each node's exploration and edge relaxation rely on dynamically updated path data, introducing sequential bottlenecks. Non-contiguous memory access patterns, driven by adjacency lists and frequent priority queue updates, degrade cache performance. In distributed implementations, frequent state synchronization imposes high communication costs. Computational divergence, where different graph regions (e.g., dense versus sparse areas) require varying levels of effort, exacerbates inefficiencies in parallel processing.
Mapping
Mapping this workload to parallel systems is challenging due to the graph's irregular structure, which complicates task partitioning and balancing. Synchronization demands for maintaining consistent shortest path estimates add bottlenecks, especially in distributed systems where communication latency is significant. Uneven computational demands across graph regions and dynamic edge weight updates hinder efficient resource allocation. Furthermore, irregular memory access and computation patterns reduce the performance of hardware architectures like GPUs, limiting the effectiveness of parallel execution.

Resources: In terms of code, we will essentially be starting from scratch; however, we will be referencing Parmentier’s paper, “Aircraft routing: complexity and algorithms,” in order to implement an initial, sequential version of our flight routing algorithm (https://cermics.enpc.fr/~parmenta/ROADEF/Rapport_MPRO_Axel_Parmentier.pdf). We will also be using an online dataset of flight routes to run our experiments on (https://www.kaggle.com/datasets/open-flights/flight-route-database), although we may make up smaller test cases to test our initial implementations. Currently, we are planning to run our experiments on the GHC machines, but it may be beneficial to run them on PSC in the future.

Goals and deliverables:
Plan to achieve:
- Initial sequential C/C++ implementation of flight routing program, which returns a sequence of flight legs (otherwise known as a flight string) for each aircraft in a fleet, given a fixed fleet size, aircraft starting positions, maintenance bases (not all airports qualify as such), and set of input flights that must be covered. Time and maintenance constraints must be kept. We believe this is achievable because we have a paper to reference.
- Parallelized versions of the above algorithm, with which to run experiments and evaluate performance trade-offs. We believe this is achievable because we’ve used these frameworks in our other assignments, and spent the semester learning how to achieve performance boosts via parallelism.
  - CUDA implementation
  - OpenMP implementation
  - MPI implementation
- Graph visualization of our program’s output on some real world data for the demo. We believe this is achievable because the aircraft routing problem is equivalent to a graph problem, and it would be cool to have a visualization of our output. It would be especially cool if we could overlay our graph with a world map, with the nodes actually located at their corresponding real-world airport locations.
Hope to achieve:
- ISPC/SIMD implementation of flight routing algorithm (placed here because we have ~4 weeks for the project, so sequential + 3 parallel approaches lets us do 1 approach per week, and this was left over)
- Explore GraphLab implementation of flight routing algorithm

Platform choice: C/C++. Cuda, OpenMP, MPI, SIMD. More specifically, we will be implementing our initial program in C/C++, then comparing the performances of different implementations using the aforementioned frameworks.

Schedule:
11/17-11/23: Initial C/C++ sequential implementation of flight routing algorithm, initial runs on real-world dataset if possible.
11/24-11/30: First parallelized version of flight routing algorithm, experiments and performance comparison with sequential version. Probably CUDA. Project milestone report.
12/1-12/7: Second parallelized version and corresponding experiments. Probably OpenMP.
12/8-12/14: Third parallelized version, corresponding experiments, and graph visualization. Final project report.
