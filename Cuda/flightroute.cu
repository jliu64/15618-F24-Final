#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <stdarg.h>

#define MAX_CONNECTIONS 64 // Maximum connections per airport
#define CODE_SIZE 4        // Airport code size (3 characters + null terminator)

// Structure to store flight data
struct Flight {
  char depart_airport[CODE_SIZE];
  int depart_day;
  int depart_time;
  char arrive_airport[CODE_SIZE];
  int arrive_day;
  int arrive_time;
};

// CUDA-compatible airport structure
struct Airport {
  int aircraft_count;
  int timestep;
  int adj_list_size;
  char code[CODE_SIZE];
  Airport *adj_list[MAX_CONNECTIONS];
};

// Function to read input file
std::vector<Flight> read_input_file(const std::string &input_filename, std::set<int> &timesteps,
                                    std::map<std::string, Airport> &airports) {
  std::ifstream fin(input_filename);
  if (!fin) {
    std::cerr << "Error: Unable to open file: " << input_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  int num_flights, num_airports;
  fin >> num_flights >> num_airports;

  std::vector<Flight> flights(num_flights);

  // Read flights
  for (int i = 0; i < num_flights; ++i) {
    fin >> flights[i].depart_airport >> flights[i].depart_day >> flights[i].depart_time
        >> flights[i].arrive_airport >> flights[i].arrive_day >> flights[i].arrive_time;

    int depart_timestep = flights[i].depart_day * 24 + flights[i].depart_time;
    int arrive_timestep = flights[i].arrive_day * 24 + flights[i].arrive_time;

    if (depart_timestep >= arrive_timestep) {
      std::cerr << "Error: Invalid flight with non-positive duration." << std::endl;
      exit(EXIT_FAILURE);
    }

    timesteps.insert(depart_timestep);
    timesteps.insert(arrive_timestep);
  }

  // Read airports
  for (int i = 0; i < num_airports; ++i) {
    std::string airport_name;
    int aircraft_count;
    fin >> airport_name >> aircraft_count;

    Airport airport = {aircraft_count, 0, 0, {0}, {nullptr}};
    strncpy(airport.code, airport_name.c_str(), CODE_SIZE - 1);
    airport.code[CODE_SIZE - 1] = '\0';
    airports[airport_name] = airport;
  }

  return flights;
}

// Create a time-expanded graph for the flights
std::map<int, std::map<std::string, Airport>> compute_equigraph(const std::vector<Flight> &flights,
                                                                const std::set<int> &timesteps,
                                                                std::map<std::string, Airport> &start_airports) {
  std::map<int, std::map<std::string, Airport>> timestep_airports;
  timestep_airports[-1] = start_airports;

  for (const auto &flight: flights) {
    int depart_timestep = flight.depart_day * 24 + flight.depart_time;
    int arrive_timestep = flight.arrive_day * 24 + flight.arrive_time;

    timestep_airports[depart_timestep][flight.depart_airport] = start_airports[flight.depart_airport];
    timestep_airports[arrive_timestep][flight.arrive_airport] = start_airports[flight.arrive_airport];

    Airport &source = timestep_airports[depart_timestep][flight.depart_airport];
    Airport &destination = timestep_airports[arrive_timestep][flight.arrive_airport];

    if (source.adj_list_size < MAX_CONNECTIONS) {
      source.adj_list[source.adj_list_size++] = &destination;
    }
  }

  return timestep_airports;
}

// Device-compatible snprintf
__device__ int snprintf_device(char *buffer, int max_len, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int len = 0;

  for (const char *p = fmt; *p != '\0' && len < max_len - 1; ++p) {
    if (*p == '%') {
      ++p;
      if (*p == 's') {
        const char *str = va_arg(args, const char*);
        while (*str && len < max_len - 1) buffer[len++] = *str++;
      } else if (*p == 'd') {
        int value = va_arg(args, int);
        char temp[16];
        int idx = 0;
        if (value < 0) {
          buffer[len++] = '-';
          value = -value;
        }
        do {
          temp[idx++] = '0' + value % 10;
          value /= 10;
        } while (value > 0);
        while (idx > 0 && len < max_len - 1) buffer[len++] = temp[--idx];
      }
    } else {
      buffer[len++] = *p;
    }
  }

  buffer[len] = '\0';
  va_end(args);
  return len;
}

// Format flight string on the device
__device__ int format_flight_string(char *buffer, int max_len, const char *code1, int time1,
                                    const char *code2 = nullptr, int time2 = -1) {
  int len = snprintf_device(buffer, max_len, "%s:%d:%d", code1, time1 / 24, time1 % 24);
  if (code2 && time2 >= 0) {
    len += snprintf_device(buffer + len, max_len - len, " -> %s:%d:%d", code2, time2 / 24, time2 % 24);
  }
  return len;
}

// Kernel to compute flight strings
__global__ void
compute_flight_string_kernel(Airport *airports, int num_airports, char *results, int *offsets, int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_airports) return;

  Airport airport = airports[idx];
  char *res = results + idx * max_len;

  if (airport.adj_list_size == 0) {
    offsets[idx] = format_flight_string(res, max_len, airport.code, airport.timestep);
  } else {
    for (int i = 0; i < airport.adj_list_size; ++i) {
      Airport *connection = airport.adj_list[i];
      offsets[idx] = format_flight_string(res, max_len, airport.code, airport.timestep,
                                          connection->code, connection->timestep);
    }
  }
}

// Host function to compute flight strings using CUDA
std::vector<std::string> compute_flight_string_cuda(const std::vector<Airport> &airports) {
  int num_airports = airports.size();
  const int max_len = 256;

  Airport *d_airports;
  char *d_results;
  int *d_offsets;

  cudaMalloc(&d_airports, num_airports * sizeof(Airport));
  cudaMemcpy(d_airports, airports.data(), num_airports * sizeof(Airport), cudaMemcpyHostToDevice);

  cudaMalloc(&d_results, num_airports * max_len * sizeof(char));
  cudaMalloc(&d_offsets, num_airports * sizeof(int));

  int threads_per_block = 256;
  int blocks = (num_airports + threads_per_block - 1) / threads_per_block;
  compute_flight_string_kernel<<<blocks, threads_per_block>>>(d_airports, num_airports, d_results, d_offsets, max_len);

  cudaDeviceSynchronize();

  std::vector<char> results(num_airports * max_len);
  std::vector<int> offsets(num_airports);

  cudaMemcpy(results.data(), d_results, num_airports * max_len * sizeof(char), cudaMemcpyDeviceToHost);
  cudaMemcpy(offsets.data(), d_offsets, num_airports * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_airports);
  cudaFree(d_results);
  cudaFree(d_offsets);

  std::vector<std::string> flight_strings(num_airports);
  for (int i = 0; i < num_airports; ++i) {
    flight_strings[i] = std::string(results.data() + i * max_len, offsets[i]);
  }

  return flight_strings;
}

// Main function
int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string input_filename = argv[1];
  std::set<int> timesteps;
  std::map<std::string, Airport> start_airports;

  std::vector<Flight> flights = read_input_file(input_filename, timesteps, start_airports);
  std::map<int, std::map<std::string, Airport>> timestep_airports = compute_equigraph(flights, timesteps,
                                                                                      start_airports);

  std::vector<Airport> airports;
  for (const auto &pair: timestep_airports) {
    for (const auto &airport_pair: pair.second) {
      airports.push_back(airport_pair.second);
    }
  }

  std::vector<std::string> flight_strings = compute_flight_string_cuda(airports);

  for (const auto &flight_string: flight_strings) {
    std::cout << flight_string << std::endl;
  }

  return 0;
}
