/**
 * Parallel Flight Routing (Sequential Version)
 * Jesse Liu (jzliu), Oscar Han (Enxuh)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <unordered_set>

#include "flightroute.h"

std::vector<Flight> read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport> &airports) {
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int num_flights, num_occupied_airports;
  fin >> num_flights >> num_occupied_airports;

  std::vector<Flight> flights(num_flights);
  //std::vector occupancy(dim_y, std::vector<int>(dim_x));

  for (auto& flight : flights) {
    fin >> flight.depart_airport >> flight.depart_day >> flight.depart_time >>
      flight.arrive_airport >> flight.arrive_day >> flight.arrive_time;
    
    // Note: Timesteps discretized into hours
    int depart_timestep = flight.depart_day * 24 + flight.depart_time;
    int arrive_timestep = flight.arrive_day * 24 + flight.arrive_time;
    if (depart_timestep >= arrive_timestep) {
      std::cout << "Flight routing is infeasible with given flights." << std::endl;
      exit(EXIT_SUCCESS);
    }
    timesteps.insert(depart_timestep);
    timesteps.insert(arrive_timestep);
  }

  for (int i = 0; i < num_occupied_airports; i++) {
    std::string airport;
    int aircraft_count;
    std::list<Airport*> adj_list;
    fin >> airport >> aircraft_count;

    airports[airport] = {aircraft_count, 0, -1, airport, adj_list};
  }

  return flights;
}

std::map<int, std::map<std::string, Airport>> compute_equigraph(
  std::vector<Flight> &flights,
  std::set<int> &timesteps,
  std::map<std::string, Airport> &start_airports
) {
  std::map<int, std::map<std::string, Airport>> timestep_airports;
  timestep_airports[-1] = start_airports;

  // Add edges for flights
  for (Flight &flight: flights) {
    int depart_timestep = flight.depart_day * 24 + flight.depart_time;
    int arrive_timestep = flight.arrive_day * 24 + flight.arrive_time;

    // Initialize departure maps if non-existent
    if (timestep_airports.find(depart_timestep) == timestep_airports.end()) {
      std::map<std::string, Airport> airports;
      timestep_airports[depart_timestep] = airports;
    }
    std::map<std::string, Airport> &depart_airports = timestep_airports[depart_timestep];
    if (depart_airports.find(flight.depart_airport) == depart_airports.end()) {
      std::list<Airport *> adj_list;
      depart_airports[flight.depart_airport] = {0, 0, depart_timestep, flight.depart_airport, adj_list};
    }
    depart_airports[flight.depart_airport].num_departures++;

    // Initialize arrival maps if non-existent
    if (timestep_airports.find(arrive_timestep) == timestep_airports.end()) {
      std::map<std::string, Airport> airports;
      timestep_airports[arrive_timestep] = airports;
    }
    std::map<std::string, Airport> &arrive_airports = timestep_airports[arrive_timestep];
    if (arrive_airports.find(flight.arrive_airport) == arrive_airports.end()) {
      std::list<Airport *> adj_list;
      arrive_airports[flight.arrive_airport] = {0, 0, arrive_timestep, flight.arrive_airport, adj_list};
    }
    arrive_airports[flight.arrive_airport].num_planes++;

    // Add edge to graph (flights)
    timestep_airports[depart_timestep][flight.depart_airport].adj_list.push_back(
      &timestep_airports[arrive_timestep][flight.arrive_airport]);
  }

  // Add edges for ground connections
  for (auto &pair: start_airports) {
    std::string code = pair.first;
    Airport &airport = pair.second;
    auto it = timesteps.begin();
    if (timestep_airports.find(*it) == timestep_airports.end()) {
      std::cerr << "Failed to initialize all nodes in graph." << std::endl;
      exit(EXIT_FAILURE);
    }
    while (timestep_airports[*it].find(code) == timestep_airports[*it].end()) {
      it++;
      if (it == timesteps.end()) {
        std::cerr << "Failed to initialize all timesteps in graph." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    std::map<std::string, Airport> &airports = timestep_airports[*it];
    airport.adj_list.push_back(&airports[code]);
    airports[code].num_planes += (airport.num_planes - airport.num_departures);
  }

  // Add edges for timestep-to-timestep ground connections
  std::set<int>::iterator iterator;
  for (iterator = timesteps.begin(); iterator != timesteps.end(); iterator++) {
    for (auto &pair: timestep_airports[*iterator]) {
      std::string code = pair.first;
      Airport &airport = pair.second;
      std::set<int>::iterator it = iterator;
      it++;
      if (it == timesteps.end()) break;
      if (timestep_airports.find(*it) == timestep_airports.end()) break;
      while (timestep_airports[*it].find(code) == timestep_airports[*it].end()) {
        it++;
        if (it == timesteps.end()) break;
      }
      if (it != timesteps.end()) {
        std::map<std::string, Airport> &airports = timestep_airports[*it];
        airport.adj_list.push_back(&airports[code]);
        airports[code].num_planes += (airport.num_planes - airport.num_departures);
      }
    }
  }

  return timestep_airports;
}

// Compute flight strings for a single airport
std::list<std::string> compute_flight_string(Airport &airport, std::unordered_set<Airport *> &visited) {
  std::list<std::string> flight_strings;

  // Check if the airport has already been visited
  if (visited.find(&airport) != visited.end()) {
    std::string flight_string =
      airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
    flight_strings.emplace_back(flight_string);
    return flight_strings; // Return an empty list to prevent cycles
  }

  // Mark the current airport as visited
  visited.insert(&airport);

  // If no adjacent airports, this is a terminal node
  if (airport.adj_list.empty()) {
    std::string flight_string =
      airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
    flight_strings.emplace_back(flight_string);
  } else {
    // Recursively compute flight strings for each connection
    for (Airport *connection: airport.adj_list) {
      std::list<std::string> new_strings = compute_flight_string(*connection, visited);
      for (const std::string &string: new_strings) {
        std::string flight_string =
          airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24) +
          " -> " + string;
        flight_strings.emplace_back(flight_string);
      }
    }
  }

  return flight_strings;
}

// Parallelized function to compute flight strings for all starting airports
std::list<std::list<std::string>> compute_flight_strings(const std::map<std::string, Airport> &start_airports) {
  std::list<std::list<std::string>> all_flight_strings;  // Changed to store list of lists

  // Prepare a thread-safe vector of results
  std::vector<std::list<std::string>> thread_results(start_airports.size());

  // Parallelize over the starting airports
  #pragma omp parallel for
  for (size_t i = 0; i < start_airports.size(); ++i) {
    auto it = std::next(start_airports.begin(), i); // Access the i-th element of the map
    const std::string &code = it->first;
    Airport &airport = const_cast<Airport &>(it->second); // Avoid const for recursive processing

    // Compute the flight strings for the current airport
    auto visited = std::unordered_set<Airport *>();
    std::list<std::string> flight_strings = compute_flight_string(airport, visited);

    // Store the result in the thread-local container
    thread_results[i] = std::move(flight_strings);
  }

  // Merge all thread-local results into the list of lists
  for (const auto &result: thread_results) {
    all_flight_strings.push_back(result);  // Add each list of strings as a separate element
  }

  return all_flight_strings;
}

int main(int argc, char *argv[]) {
  if (argc <= 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <num_threads>" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string input_filename = argv[1];
  size_t num_threads;
  try {
    num_threads = std::stoi(argv[2]);
    if (num_threads <= 0) {
      throw std::invalid_argument("Number of threads must be greater than zero.");
    }
  } catch (const std::exception &e) {
    std::cerr << "Invalid number of threads: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Input file: " << input_filename << std::endl;
  std::cout << "Number of threads: " << num_threads << std::endl;
  omp_set_num_threads(num_threads);

  // Read input data
  std::set<int> timesteps;
  std::map<std::string, Airport> start_airports;
  std::vector<Flight> flights = read_input_file(input_filename, timesteps, start_airports);

  // Start timing the rest of the program
  auto total_start = std::chrono::high_resolution_clock::now();

  // Build the time-expanded flight graph (equigraph)
  auto start_graph = std::chrono::high_resolution_clock::now();
  auto timestep_airports = compute_equigraph(flights, timesteps, start_airports);
  auto end_graph = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken to build graph: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(end_graph - start_graph).count()
            << std::fixed << std::setprecision(10) << " s" << std::endl;

  // Validate the graph for feasibility
  auto start_validation = std::chrono::high_resolution_clock::now();
  for (auto it = timesteps.begin(); it != timesteps.end(); ++it) {
    for (auto &pair: timestep_airports[*it]) {
      Airport &airport = pair.second;
      if (airport.num_planes < airport.num_departures) {
        std::cout << "Flight routing is infeasible with given flights." << std::endl;
        exit(EXIT_SUCCESS);
      }
    }
  }
  auto end_validation = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken for validation: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(end_validation - start_validation).count()
            << std::fixed << std::setprecision(10) << " s" << std::endl;

  // Compute the flight strings for all starting airports
  auto start_compute = std::chrono::high_resolution_clock::now();
  std::list<std::list<std::string>> flight_strings = compute_flight_strings(start_airports);
  auto end_compute = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken to compute flight strings: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(end_compute - start_compute).count()
            << std::fixed << std::setprecision(10) << " s" << std::endl;

  // End timing the rest of the program
  auto total_end = std::chrono::high_resolution_clock::now();
  std::cout << "Total time taken (excluding input read): "
            << std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count()
            << std::fixed << std::setprecision(10) << " s" << std::endl;

  // Output the flight strings
  for (std::list<std::string> str_list : flight_strings)
    for (std::string str : str_list)
      std::cout << str << std::endl;

  return 0;
}
