#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <unordered_map>

#define MAX_FLIGHTS 100
#define MAX_PLANES 10
#define MAX_AIRPORTS 10
#define MAX_STRING_SIZE 10

// Flight structure
typedef struct {
  char source[MAX_STRING_SIZE];
  char destination[MAX_STRING_SIZE];
  char airline[MAX_STRING_SIZE];
  int stops;
  char equipment[MAX_STRING_SIZE];
} Flight;

// Plane structure
typedef struct {
  int id;
  int num_flights;
  char assigned_sources[MAX_FLIGHTS][MAX_STRING_SIZE];
  char assigned_destinations[MAX_FLIGHTS][MAX_STRING_SIZE];
} Plane;

// Device-compatible strcmp function
__device__ int strcmp_device(const char *str1, const char *str2) {
  while (*str1 && (*str1 == *str2)) {
    str1++;
    str2++;
  }
  return *(unsigned char *) str1 - *(unsigned char *) str2;
}

__device__ void strncpy_device(char *dest, const char *src, size_t n) {
  size_t i = 0;
  while (i < n - 1 && src[i] != '\0') {
    dest[i] = src[i];
    i++;
  }
  dest[i] = '\0'; // Null-terminate the destination string
}

// Kernel to initialize planes
__global__ void initialize_planes(
  Plane *planes, int *initial_positions,
  char *airports[], int num_planes, int num_airports) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_planes) {
    planes[idx].id = idx;
    planes[idx].num_flights = 0;

    // Assign plane to an initial airport based on initial_positions
    for (int i = 0; i < num_airports; i++) {
      if (initial_positions[i] > 0) {
        initial_positions[i]--;
        strncpy_device(planes[idx].assigned_sources[0], airports[i], MAX_STRING_SIZE);
        break;
      }
    }
  }
}

// Kernel to assign flights to planes
__global__ void assign_flights(
  const Flight *flights, int num_flights,
  Plane *planes, int num_planes,
  int *final_positions, int num_airports,
  char *airports[]) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_planes) return;

  Plane *plane = &planes[idx];

  // Each thread works on one plane, starting from its initial position
  char *start_airport = plane->assigned_sources[0];

  // Create a visited set to avoid revisiting airports
  bool visited[MAX_AIRPORTS] = {false};
  int visited_index = -1;
  for (int j = 0; j < num_airports; j++) {
    if (strcmp_device(start_airport, airports[j]) == 0) {
      visited_index = j;
      visited[j] = true;
      break;
    }
  }

  // If the initial position is invalid, skip processing
  if (visited_index == -1) return;

  // Traverse flights in BFS-style to assign to the plane
  for (int i = 0; i < num_flights; i++) {
    const Flight &current_flight = flights[i];

    // Check if the flight's source matches the plane's current position
    char *last_airport = plane->num_flights == 0
                         ? start_airport
                         : plane->assigned_destinations[plane->num_flights - 1];

    if (strcmp_device(current_flight.source, last_airport) == 0 && !visited[visited_index]) {
      // Assign the flight to the plane
      strncpy_device(plane->assigned_sources[plane->num_flights], current_flight.source, MAX_STRING_SIZE);
      strncpy_device(plane->assigned_destinations[plane->num_flights], current_flight.destination, MAX_STRING_SIZE);
      plane->num_flights++;

      // Mark the destination as visited
      for (int j = 0; j < num_airports; j++) {
        if (strcmp_device(current_flight.destination, airports[j]) == 0) {
          visited[j] = true;
          visited_index = j;
          break;
        }
      }

      // Update final positions for the last destination
      if (plane->num_flights == num_flights || i == num_flights - 1) {
        atomicAdd(&final_positions[visited_index], 1);
      }
    }
  }
}

// Validation function using STL
bool validate_assignments(const std::unordered_map<std::string, int> &final_positions,
                          const std::unordered_map<std::string, int> &final_requirements) {
  bool all_met = true;
  for (const auto &req: final_requirements) {
    if (final_positions.at(req.first) < req.second) {
      all_met = false;
      std::cerr << "Requirement not met at airport: " << req.first << " ("
                << final_positions.at(req.first) << "/" << req.second << " planes available)\n";
    }
  }
  return all_met;
}

int main() {
  // Input Data
  const int num_planes = 2;
  const int num_flights = 3;
  const int num_airports = 3;

  // Initial Aircraft Positions
  int initial_positions[] = {2, 0, 0}; // Example: 2 planes at LAX, 0 at JFK and ATL
  const std::unordered_map<std::string, int> initial_positions_map = {
    {"LAX", 2},
    {"JFK", 0},
    {"ATL", 0}};

  // Final Aircraft Requirements
  const int final_requirements_arr[] = {1, 1, 0};
  const std::unordered_map<std::string, int> final_requirements_map = {
    {"LAX", 1},
    {"JFK", 1},
    {"ATL", 0}};
  const char *airports[] = {"LAX", "JFK", "ATL"};

  // Flight Data
  Flight flights[] = {
    {"LAX", "JFK", "AA", 0, "738"},
    {"JFK", "ATL", "AA", 0, "738"},
    {"ATL", "LAX", "AA", 0, "738"}};

  // Allocate device memory
  Flight *d_flights;
  Plane *d_planes;
  int *d_initial_positions, *d_final_positions;
  char **d_airports;

  cudaMalloc((void **) &d_flights, num_flights * sizeof(Flight));
  cudaMalloc((void **) &d_planes, num_planes * sizeof(Plane));
  cudaMalloc((void **) &d_initial_positions, num_airports * sizeof(int));
  cudaMalloc((void **) &d_final_positions, num_airports * sizeof(int));
  cudaMalloc((void **) &d_airports, num_airports * sizeof(char *));

  // Copy data to device
  cudaMemcpy(d_flights, flights, num_flights * sizeof(Flight), cudaMemcpyHostToDevice);
  cudaMemcpy(d_initial_positions, initial_positions, num_airports * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_final_positions, final_requirements_arr, num_airports * sizeof(int), cudaMemcpyHostToDevice);

  // Initialize planes
  int threads_per_block = 32;
  int blocks = (num_planes + threads_per_block - 1) / threads_per_block;
  initialize_planes<<<blocks, threads_per_block>>>(d_planes, d_initial_positions, d_airports, num_planes, num_airports);
  cudaDeviceSynchronize();

  // Assign flights
  assign_flights<<<blocks, threads_per_block>>>(d_flights, num_flights, d_planes, num_planes, d_final_positions,
                                                num_airports, d_airports);
  cudaDeviceSynchronize();

  // Copy results back to host
  Plane planes[MAX_PLANES];
  int final_positions_arr[MAX_AIRPORTS] = {0};
  cudaMemcpy(planes, d_planes, num_planes * sizeof(Plane), cudaMemcpyDeviceToHost);
  cudaMemcpy(final_positions_arr, d_final_positions, num_airports * sizeof(int), cudaMemcpyDeviceToHost);

  // Convert results to STL maps for validation
  std::unordered_map<std::string, int> final_positions_map;
  for (int i = 0; i < num_airports; i++) {
    final_positions_map[airports[i]] = final_positions_arr[i];
  }

  // Validate assignments
  bool all_met = validate_assignments(final_positions_map, final_requirements_map);

  if (all_met) {
    std::cout << "All requirements met.\n";
  } else {
    std::cout << "Some requirements not met.\n";
  }

  // Free device memory
  cudaFree(d_flights);
  cudaFree(d_planes);
  cudaFree(d_initial_positions);
  cudaFree(d_final_positions);
  cudaFree(d_airports);

  return 0;
}
