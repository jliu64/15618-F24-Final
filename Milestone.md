### Summary:
We have successfully completed the implementation of our sequential flight routing algorithm (equivalent to a minimum cost network path partition problem). This implementation provides a solid foundation for benchmarking and will help identify bottlenecks to address in the upcoming parallelization phase.
We also evaluated this algorithm on the Open Flights Route Database (https://www.kaggle.com/datasets/open-flights/flight-route-database) from Kaggle, which proved to be inclusive and reliable. The dataset covers diverse routing characteristics, such as varying route lengths, node densities, and edge complexities. It showed no signs of incompleteness or inconsistency, making it suitable for validating our implementations.

### Goals:
We have successfully achieved the first goal outlined in our proposal: completing the initial sequential C/C++ implementation of the flight routing program. The program generates a sequence of flight legs for each aircraft in a fleet while respecting constraints such as fleet size, starting positions, maintenance bases, and time. This implementation has been thoroughly tested using real-world data from the Open Flights Route Database, confirming its accuracy and reliability in handling various routing scenarios. The program aligns well with our goal of adhering to both time and maintenance constraints, demonstrating its practicality.

We are also making significant progress toward the second goal. While we have not yet completed all parallelized implementations, we have finished developing the CUDA version of the algorithm. Although it has not been fully tested, this represents a major milestone since CUDA required the most extensive coding and debugging effort. Successfully implementing CUDA has also allowed us to establish a more extensible infrastructure, streamlining the integration of the remaining OpenMP and MPI implementations. These are comparatively less complex and will be easier to integrate within the refined codebase. Given this progress, we are confident in our ability to complete all deliverables, including the remaining parallelized versions, by the final deadline.
We remain confident in achieving our third goal: providing a graph visualization of our program's output using real-world data for the demo. Leveraging existing tools like FlightMapper.io, which offers interactive 2D maps with flight data, we can effectively overlay our routing graphs onto a world map, accurately positioning nodes at their corresponding real-world airport locations. This approach will enhance the clarity and impact of our visualizations during the poster session.

### Poster:
At the poster session, we plan to present a live demo of our flight routing program. The demo will showcase the program's ability to generate optimized flight strings for a given fleet, visualized through a graph overlaid on a 2D map. This demonstration will highlight both the functionality of our sequential and parallel implementations and the efficiency improvements achieved through different parallelization strategies.

### Preliminary Results:
We don’t have preliminary results to share at this time, as we haven’t finished testing the first parallelized version of our algorithm (i.e. the CUDA implementation). When we’ve implemented and tested our parallel implementations, we’ll be able to share data on speedup relative to the sequential version, among other metrics.

### Concerning Issues:
We don’t have any especially concerning issues at this time; we pretty much just need to put the remaining work in to finish the project.

### Updated Schedule:
See README.md
