#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_FLIGHTS 100
#define MAX_PLANES 10
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

// Kernel to initialize planes
__global__ void initialize_planes(Plane *planes, int num_planes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_planes) {
    planes[idx].id = idx;
    planes[idx].num_flights = 0;
  }
}

// Kernel to assign flights to planes
__global__ void assign_flights(
  const Flight *flights, int num_flights,
  Plane *planes, int num_planes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_planes) return;

  // Each thread processes one plane
  for (int i = idx; i < num_flights; i += num_planes) {
    int flight_idx = planes[idx].num_flights;
    if (flight_idx < MAX_FLIGHTS) {
      // Copy source and destination strings
      for (int j = 0; j < MAX_STRING_SIZE; j++) {
        planes[idx].assigned_sources[flight_idx][j] = flights[i].source[j];
        planes[idx].assigned_destinations[flight_idx][j] = flights[i].destination[j];
      }

      // Null-terminate strings
      planes[idx].assigned_sources[flight_idx][MAX_STRING_SIZE - 1] = '\0';
      planes[idx].assigned_destinations[flight_idx][MAX_STRING_SIZE - 1] = '\0';

      planes[idx].num_flights++;
    }
  }
}

// Host function to print plane assignments
void print_plane_assignments(Plane *planes, int num_planes) {
  for (int i = 0; i < num_planes; i++) {
    printf("Plane_%d Assignments:\n", planes[i].id);
    for (int j = 0; j < planes[i].num_flights; j++) {
      printf("  %s -> %s\n", planes[i].assigned_sources[j], planes[i].assigned_destinations[j]);
    }
  }
}

int main() {
  // Number of planes and flights
  const int num_planes = 3;
  const int num_flights = 5;

  // Host-side flight data
  Flight flights[MAX_FLIGHTS] = {
    {"LAX", "JFK", "AA", 0, "738"},
    {"JFK", "ATL", "AA", 0, "738"},
    {"ATL", "LAX", "AA", 0, "738"},
    {"LAX", "SFO", "UA", 0, "320"},
    {"SFO", "SEA", "UA", 0, "320"}
  };

  // Allocate device memory
  Flight *d_flights;
  Plane *d_planes;
  cudaMalloc((void **) &d_flights, num_flights * sizeof(Flight));
  cudaMalloc((void **) &d_planes, num_planes * sizeof(Plane));

  // Copy flight data to the device
  cudaMemcpy(d_flights, flights, num_flights * sizeof(Flight), cudaMemcpyHostToDevice);

  // Launch kernel to initialize planes
  int threads_per_block = 32;
  int blocks = (num_planes + threads_per_block - 1) / threads_per_block;
  initialize_planes<<<blocks, threads_per_block>>>(d_planes, num_planes);
  cudaDeviceSynchronize();

  // Launch kernel to assign flights to planes
  assign_flights<<<blocks, threads_per_block>>>(d_flights, num_flights, d_planes, num_planes);
  cudaDeviceSynchronize();

  // Copy results back to the host
  Plane planes[MAX_PLANES];
  cudaMemcpy(planes, d_planes, num_planes * sizeof(Plane), cudaMemcpyDeviceToHost);

  // Print results
  print_plane_assignments(planes, num_planes);

  // Free device memory
  cudaFree(d_flights);
  cudaFree(d_planes);

  return 0;
}
