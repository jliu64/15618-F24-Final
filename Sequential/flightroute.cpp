/**
 * Parallel Flight Routing (Sequential Version)
 * Jesse Liu (jzliu), Oscar Han (Enxuh)
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include <unistd.h>
#include <omp.h>
#include <stdlib.h>
#include <cmath>
#include <climits>
#include <utility>
/*
void print_stats(const std::vector<std::vector<int>>& occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto& row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}
*/
/*
void write_output(const std::vector<Wire>& wires, const int num_wires, const std::vector<std::vector<int>>& occupancy, const int dim_x, const int dim_y, const int num_threads, std::string input_filename) {
  if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
    input_filename.resize(std::size(input_filename) - 4);
  }

  const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(num_threads) + ".txt";
  const std::string wires_filename = input_filename + "_wires_" + std::to_string(num_threads) + ".txt";

  std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_occupancy << dim_x << ' ' << dim_y << '\n';
  for (const auto& row : occupancy) {
    for (const int count : row) {
      out_occupancy << count << ' ';
    }
    out_occupancy << '\n';
  }

  out_occupancy.close();

  std::ofstream out_wires(wires_filename, std::fstream:: out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

  for (const auto& [start_x, start_y, end_x, end_y, bend1_x, bend1_y] : wires) {
    out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

    if (start_y == bend1_y) {
    // first bend was horizontal

      if (end_x != bend1_x) {
        // two bends

        out_wires << bend1_x << ' ' << end_y << ' ';
      }
    } else if (start_x == bend1_x) {
      // first bend was vertical

      if(end_y != bend1_y) {
        // two bends

        out_wires << end_x << ' ' << bend1_y << ' ';
      }
    }
    out_wires << end_x << ' ' << end_y << '\n';
  }

  out_wires.close();
}
*/

int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();

  std::string input_filename;
  int num_threads = 0;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;

  int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 'n':
        num_threads = atoi(optarg);
        break;
      case 'p':
        SA_prob = atof(optarg);
        break;
      case 'i':
        SA_iters = atoi(optarg);
        break;
      case 'm':
        parallel_mode = *optarg;
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || num_threads <= 0 || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Number of threads: " << num_threads << '\n';
  std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
  std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
  std::cout << "Input file: " << input_filename << '\n';
  std::cout << "Parallel mode: " << parallel_mode << '\n';
  std::cout << "Batch size: " << batch_size << '\n';

  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int dim_x, dim_y;
  int num_wires;

  /* Read the grid dimension and wire information from file */
  fin >> dim_x >> dim_y >> num_wires;

  std::vector<Wire> wires(num_wires);
  std::vector occupancy(dim_y, std::vector<int>(dim_x));

  for (auto& wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.bend1_x = -1;
    wire.bend1_y = -1;
  }

  /* Initialize any additional data structures needed in the algorithm */
  omp_set_num_threads(num_threads);
  std::vector locks(dim_y, std::vector<omp_lock_t>(dim_x));
  std::vector<std::pair<int, int>> mins(num_wires);
  if (parallel_mode == 'A') {
    for (auto& row : locks) {
      for (omp_lock_t lock : row) {
        omp_init_lock(&lock);
      }
    }
  }

  const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
  std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

  const auto compute_start = std::chrono::steady_clock::now();

  /** 
   * Implement the wire routing algorithm here
   * Feel free to structure the algorithm into different functions
   * Don't use global variables.
   * Use OpenMP to parallelize the algorithm. 
   */
  if (parallel_mode == 'W') {
    for (int iter = 0; iter < SA_iters; iter++) {
      for (auto& wire : wires) {
        // If wire is completely horizontal or vertical, there's only one valid path
        if (wire.start_x == wire.end_x || wire.start_y == wire.end_y) {
          if (wire.bend1_x == -1 && wire.bend1_y == -1) { // Wire shouldn't already have a path
            wire.bend1_x = wire.start_x;
            wire.bend1_y = wire.start_y;
            add_occupancy(wire, occupancy);
          }

          continue;
        }

        // With probability P, choose a random path
        double r = ((double) rand() / (RAND_MAX));
        if (r < SA_prob) {
          int hori_or_vert = rand() % 2;
          int x, y;
          if (hori_or_vert == 0) { // Path starts horizontally
            x = std::max(1, rand() % std::abs(wire.start_x - wire.end_x));
            x = wire.start_x <= wire.end_x ? wire.start_x + x : wire.start_x - x;
            y = wire.start_y;
          } else {
            y = std::max(1, rand() % std::abs(wire.start_y - wire.end_y));
            y = wire.start_y <= wire.end_y ? wire.start_y + y : wire.start_y - y;
            x = wire.start_x;
          }
          if (wire.bend1_x != -1 && wire.bend1_y != -1) { // Wire already has a path
            clear_occupancy(wire, occupancy);
          }
          wire.bend1_x = x;
          wire.bend1_y = y;
          add_occupancy(wire, occupancy);
          continue;
        }

        // Compute minimum path for given wire
        int min_x = -1;
        int min_y = -1;
        int min_max = INT_MAX;

        // If wire is already at minimum, don't bother
        if (wire.bend1_x != -1 && wire.bend1_y != -1) {
          min_max = compute_theory_max(wire, occupancy, wire.bend1_x, wire.bend1_y, min_max);
          min_x = wire.bend1_x;
          min_y = wire.bend1_y;
          if (min_max == 1) continue;
        }

        // Check paths that start horizontally
        if (wire.start_x <= wire.end_x) {
          #pragma omp parallel for schedule(dynamic)
            for (int x = wire.start_x + 1; x <= wire.end_x; x++) {
              if (min_max == 1) continue;
              int max = compute_theory_max(wire, occupancy, x, wire.start_y, min_max);
              bool match = false;
              {
                #pragma omp critical
                  if (max < min_max) {
                    min_max = max;
                    min_x = x;
                    min_y = wire.start_y;
                  } else if (max == min_max) {
                    match = true;
                  }
              }
              if (match) {
                int old_cost = compute_theory_cost(wire, occupancy, min_x, min_y);
                int new_cost = compute_theory_cost(wire, occupancy, x, wire.start_y);
                #pragma omp critical
                  if (old_cost < new_cost) {
                    min_x = x;
                    min_y = wire.start_y;
                  }
              }
            }
        } else {
          #pragma omp parallel for schedule(dynamic)
            for (int x = wire.start_x - 1; x >= wire.end_x; x--) {
              if (min_max == 1) continue;
              int max = compute_theory_max(wire, occupancy, x, wire.start_y, min_max);
              bool match = false;
              {
                #pragma omp critical
                  if (max < min_max) {
                    min_max = max;
                    min_x = x;
                    min_y = wire.start_y;
                  } else if (max == min_max) {
                    match = true;
                  }
              }
              if (match) {
                int old_cost = compute_theory_cost(wire, occupancy, min_x, min_y);
                int new_cost = compute_theory_cost(wire, occupancy, x, wire.start_y);
                #pragma omp critical
                  if (old_cost < new_cost) {
                    min_x = x;
                    min_y = wire.start_y;
                  }
              }
            }
        }

        // Check paths that start vertically
        if (wire.start_y <= wire.end_y) {
          #pragma omp parallel for schedule(dynamic)
            for (int y = wire.start_y + 1; y <= wire.end_y; y++) {
              if (min_max == 1) continue;
              int max = compute_theory_max(wire, occupancy, wire.start_x, y, min_max);
              bool match = false;
              {
                #pragma omp critical
                  if (max < min_max) {
                    min_max = max;
                    min_x = wire.start_x;
                    min_y = y;
                  } else if (max == min_max) {
                    match = true;
                  }
              }
              if (match) {
                int old_cost = compute_theory_cost(wire, occupancy, min_x, min_y);
                int new_cost = compute_theory_cost(wire, occupancy, wire.start_x, y);
                #pragma omp critical
                  if (old_cost < new_cost) {
                    min_x = wire.start_x;
                    min_y = y;
                  }
              }
            }
        } else {
          #pragma omp parallel for schedule(dynamic)
            for (int y = wire.start_y - 1; y >= wire.end_y; y--) {
              if (min_max == 1) continue;
              int max = compute_theory_max(wire, occupancy, wire.start_x, y, min_max);
              bool match = false;
              {
                #pragma omp critical
                  if (max < min_max) {
                    min_max = max;
                    min_x = wire.start_x;
                    min_y = y;
                  } else if (max == min_max) {
                    match = true;
                  }
              }
              if (match) {
                int old_cost = compute_theory_cost(wire, occupancy, min_x, min_y);
                int new_cost = compute_theory_cost(wire, occupancy, wire.start_x, y);
                #pragma omp critical
                  if (old_cost < new_cost) {
                    min_x = wire.start_x;
                    min_y = y;
                  }
              }
            }
        }

        if (wire.bend1_x != min_x && wire.bend1_y != min_y) {
          if (wire.bend1_x != -1 && wire.bend1_y != -1) { // Wire already has a path
            clear_occupancy(wire, occupancy);
          }
          wire.bend1_x = min_x;
          wire.bend1_y = min_y;
          add_occupancy(wire, occupancy);
        }
      }
    }
  } else { // Parallelize across wires
    for (int iter = 0; iter < SA_iters; iter++) {
      for (int i = 0; i < num_wires; i += batch_size) {
        // Compute minimum paths for each wire in the batch in parallel
        int batch_end = std::min(i + batch_size, num_wires);

        #pragma omp parallel for schedule(dynamic)
          for (int j = i; j < batch_end; j++) {
            Wire& wire = wires[j];

            // If wire is completely horizontal or vertical, there's only one valid path
            if (wire.start_x == wire.end_x || wire.start_y == wire.end_y) {
              if (wire.bend1_x == -1 && wire.bend1_y == -1) { // Wire shouldn't already have a path
                mins[j].first = wire.start_x;
                mins[j].second = wire.start_y;
              }

              continue;
            }

            // With probability P, choose a random path
            double r = ((double) rand() / (RAND_MAX));
            if (r < SA_prob) {
              int hori_or_vert = rand() % 2;
              int x, y;
              if (hori_or_vert == 0) { // Path starts horizontally
                x = std::max(1, rand() % std::abs(wire.start_x - wire.end_x));
                x = wire.start_x <= wire.end_x ? wire.start_x + x : wire.start_x - x;
                y = wire.start_y;
              } else {
                y = std::max(1, rand() % std::abs(wire.start_y - wire.end_y));
                y = wire.start_y <= wire.end_y ? wire.start_y + y : wire.start_y - y;
                x = wire.start_x;
              }
              
              mins[j].first = x;
              mins[j].second = y;
              continue;
            }

            // Compute minimum path for given wire
            int min_max = INT_MAX;

            // If wire is already at minimum, don't bother
            if (wire.bend1_x != -1 && wire.bend1_y != -1) {
              min_max = compute_theory_max(wire, occupancy, wire.bend1_x, wire.bend1_y, min_max);
              mins[j].first = wire.bend1_x;
              mins[j].second = wire.bend1_y;
              if (min_max == 1) continue;
            }

            // Check paths that start horizontally
            if (wire.start_x <= wire.end_x) {
              for (int x = wire.start_x + 1; x <= wire.end_x; x++) {
                if (min_max == 1) continue;
                int max = compute_theory_max(wire, occupancy, x, wire.start_y, min_max);
                bool match = false;
                {
                  if (max < min_max) {
                    min_max = max;
                    mins[j].first = x;
                    mins[j].second = wire.start_y;
                  } else if (max == min_max) {
                    match = true;
                  }
                }
                if (match) {
                  int old_cost = compute_theory_cost(wire, occupancy, mins[j].first, mins[j].second);
                  int new_cost = compute_theory_cost(wire, occupancy, x, wire.start_y);
                  if (old_cost < new_cost) {
                    mins[j].first = x;
                    mins[j].second = wire.start_y;
                  }
                }
              }
            } else {
              for (int x = wire.start_x - 1; x >= wire.end_x; x--) {
                if (min_max == 1) continue;
                int max = compute_theory_max(wire, occupancy, x, wire.start_y, min_max);
                bool match = false;
                {
                  if (max < min_max) {
                    min_max = max;
                    mins[j].first = x;
                    mins[j].second = wire.start_y;
                  } else if (max == min_max) {
                    match = true;
                  }
                }
                if (match) {
                  int old_cost = compute_theory_cost(wire, occupancy, mins[j].first, mins[j].second);
                  int new_cost = compute_theory_cost(wire, occupancy, x, wire.start_y);
                  if (old_cost < new_cost) {
                    mins[j].first = x;
                    mins[j].second = wire.start_y;
                  }
                }
              }
            }

            // Check paths that start vertically
            if (wire.start_y <= wire.end_y) {
              for (int y = wire.start_y + 1; y <= wire.end_y; y++) {
                if (min_max == 1) continue;
                int max = compute_theory_max(wire, occupancy, wire.start_x, y, min_max);
                bool match = false;
                {
                  if (max < min_max) {
                    min_max = max;
                    mins[j].first = wire.start_x;
                    mins[j].second = y;
                  } else if (max == min_max) {
                    match = true;
                  }
                }
                if (match) {
                  int old_cost = compute_theory_cost(wire, occupancy, mins[j].first, mins[j].second);
                  int new_cost = compute_theory_cost(wire, occupancy, wire.start_x, y);
                  if (old_cost < new_cost) {
                    mins[j].first = wire.start_x;
                    mins[j].second = y;
                  }
                }
              }
            } else {
              for (int y = wire.start_y - 1; y >= wire.end_y; y--) {
                if (min_max == 1) continue;
                int max = compute_theory_max(wire, occupancy, wire.start_x, y, min_max);
                bool match = false;
                {
                  if (max < min_max) {
                    min_max = max;
                    mins[j].first = wire.start_x;
                    mins[j].second = y;
                  } else if (max == min_max) {
                    match = true;
                  }
                }
                if (match) {
                  int old_cost = compute_theory_cost(wire, occupancy, mins[j].first, mins[j].second);
                  int new_cost = compute_theory_cost(wire, occupancy, wire.start_x, y);
                  if (old_cost < new_cost) {
                    mins[j].first = wire.start_x;
                    mins[j].second = y;
                  }
                }
              }
            }
          }
        // Modify the occupancy matrix
        #pragma omp parallel for schedule(dynamic)
          for (int j = i; j < batch_end; j++) {
            Wire& wire = wires[j];
            if (wire.bend1_x != mins[j].first && wire.bend1_y != mins[j].second) {
              if (wire.bend1_x != -1 && wire.bend1_y != -1) { // Wire already has a path
                clear_occupancy_sync(wire, occupancy, locks);
              }
              wire.bend1_x = mins[j].first;
              wire.bend1_y = mins[j].second;
              add_occupancy_sync(wire, occupancy, locks);
            }
          }
      }
    }
  }

  const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
  std::cout << "Computation time (sec): " << compute_time << '\n';
  std::cout << "Total time (sec): " << init_time + compute_time << '\n';

  /* Write wires and occupancy matrix to files */

  print_stats(occupancy);
  write_output(wires, num_wires, occupancy, dim_x, dim_y, num_threads, input_filename);
}

validate_wire_t Wire::to_validate_format(void) const {
  /* TODO(student): Implement this if you want to use the wr_checker. */
  /* See wireroute.h for details on validate_wire_t. */
  throw std::logic_error("to_validate_format not implemented.");
}
