#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <omp.h>

// Define a flight structure
struct flight {
  std::string source;
  std::string destination;
  std::string airline;
  int stops;
  std::string equipment;
};

// Define a plane structure
struct plane {
  int id;
  std::vector <std::string> assigned_flights;
};

// Build the flight graph
std::unordered_map <std::string, std::vector<flight>> build_flight_graph(const std::vector <flight> &flights) {
  std::unordered_map <std::string, std::vector<flight>> graph;
  for (const auto &f: flights) {
    graph[f.source].push_back(f);
  }
  return graph;
}

// Assign flights to planes using OpenMP
std::unordered_map<int, plane> assign_flights(
  const std::unordered_map <std::string, std::vector<flight>> &flight_graph,
  const std::unordered_map<std::string, int> &initial_positions,
  const std::unordered_map<std::string, int> &final_requirements,
  int fleet_size) {

  std::unordered_map<int, plane> plane_assignments;
  std::vector<int> plane_used(fleet_size, 0);
  std::unordered_map<std::string, int> final_positions;

  // Initialize planes
  #pragma omp parallel for
  for (int i = 0; i < fleet_size; ++i) {
    plane_assignments[i] = plane{i, {}};
  }

  // Convert the unordered_map to a vector for OpenMP compatibility
  std::vector <std::pair<std::string, int>> positions_vec(initial_positions.begin(), initial_positions.end());

  // Parallel loop over the vector of initial positions
  #pragma omp parallel for
  for (size_t idx = 0; idx < positions_vec.size(); ++idx) {
    const auto &entry = positions_vec[idx];
    std::string start_airport = entry.first;
    int planes_available = entry.second;

    for (int p = 0; p < planes_available; ++p) {
      // Find an available plane
      int plane_id = -1;
      #pragma omp critical
      {
        for (int i = 0; i < fleet_size; ++i) {
          if (!plane_used[i]) {
            plane_id = i;
            plane_used[i] = 1;
            break;
          }
        }
      }

      if (plane_id == -1) continue; // No plane available

      // BFS to assign flights to the plane
      std::unordered_set <std::string> visited;
      std::queue <std::string> to_visit;
      to_visit.push(start_airport);

      while (!to_visit.empty()) {
        std::string current_airport = to_visit.front();
        to_visit.pop();

        if (visited.count(current_airport)) continue;
        visited.insert(current_airport);

        for (const auto &f: flight_graph.at(current_airport)) {
          if (!visited.count(f.destination)) {
            #pragma omp critical
            {
              plane_assignments[plane_id].assigned_flights.push_back(f.source + " -> " + f.destination);
            }
            to_visit.push(f.destination);

            // Track the final position of the plane
            #pragma omp critical
            {
              final_positions[f.destination]++;
            }
          }
        }
      }
    }
  }

  // Check final requirements
  bool requirements_met = true;
  for (const auto &req: final_requirements) {
    if (final_positions[req.first] < req.second) {
      requirements_met = false;
      std::cerr << "Requirement not met at airport: " << req.first << "\n";
    }
  }

  if (requirements_met) {
    std::cout << "All final requirements met.\n";
  } else {
    std::cerr << "Some requirements were not met.\n";
  }

  return plane_assignments;
}

int main() {
  int num_threads = 4;
  omp_set_num_threads(num_threads);

  // Dummy flight data
  std::vector <flight> flights = {
    {"LAX", "JFK", "AA", 0, "738"},
    {"JFK", "ATL", "AA", 0, "738"},
    {"ATL", "LAX", "AA", 0, "738"}
  };

  // Initial aircraft positions
  std::unordered_map<std::string, int> initial_positions = {
    {"LAX", 2},
    {"JFK", 0},
    {"ATL", 0}
  };

  // Final aircraft requirements
  std::unordered_map<std::string, int> final_requirements = {
    {"LAX", 1},
    {"JFK", 1},
    {"ATL", 0}
  };

  // Fleet size
  int fleet_size = 2;

  // Build the flight graph
  std::unordered_map <std::string, std::vector<flight>> flight_graph = build_flight_graph(flights);

  // Assign flights to planes
  std::unordered_map<int, plane> plane_assignments = assign_flights(
    flight_graph, initial_positions, final_requirements, fleet_size);

  // Output results
  for (const auto &entry: plane_assignments) {
    const auto &p = entry.second;
    std::cout << "Plane_" << p.id << " Assignments:\n";
    for (const auto &route: p.assigned_flights) {
      std::cout << "  " << route << "\n";
    }
  }

  return 0;
}
