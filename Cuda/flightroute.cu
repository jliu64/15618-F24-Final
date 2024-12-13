#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <string>
#include <omp.h>

// Include necessary structures and types
#include "flightroute.h"

// Read input file and construct initial flight and airport data
std::vector<Flight>
read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport> &airports) {
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int num_flights, num_occupied_airports;
  fin >> num_flights >> num_occupied_airports;

  std::vector<Flight> flights(num_flights);

  for (int i = 0; i < num_flights; ++i) {
    fin >> flights[i].depart_airport >> flights[i].depart_day >> flights[i].depart_time >>
        flights[i].arrive_airport >> flights[i].arrive_day >> flights[i].arrive_time;

    // Convert departure and arrival times to timesteps (discretized into hours)
    int depart_timestep = flights[i].depart_day * 24 + flights[i].depart_time;
    int arrive_timestep = flights[i].arrive_day * 24 + flights[i].arrive_time;

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
    std::list<Airport *> adj_list;
    fin >> airport >> aircraft_count;

    airports[airport] = {aircraft_count, 0, -1, airport, adj_list};
  }

  return flights;
}

// Build the equigraph (time-expanded flight graph)
std::map<int, std::map<std::string, Airport>> compute_equigraph(std::vector<Flight> &flights, std::set<int> &timesteps,
                                                                std::map<std::string, Airport> &start_airports) {
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

    // Add edge to graph
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

  std::set<int>::iterator iterator;
  for (iterator = timesteps.begin(); iterator != timesteps.end(); iterator++) {
    for (auto &pair: timestep_airports[*iterator]) {
      std::string code = pair.first;
      Airport &airport = pair.second;
      std::set<int>::iterator it = iterator;
      it++;
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

// Compute flight chains recursively (parallelized)
std::list<std::string> compute_flight_string(Airport &airport) {
  std::list<std::string> flight_strings;

  if (airport.adj_list.empty()) {
    std::string flight_string =
      airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
    flight_strings.emplace_back(flight_string);
    return flight_strings;
  }

  std::vector<Airport *> adj_vector(airport.adj_list.begin(), airport.adj_list.end());

  #pragma omp parallel
  {
    std::list<std::string> local_flight_strings;

    #pragma omp for nowait
    for (size_t i = 0; i < adj_vector.size(); ++i) {
      Airport *connection = adj_vector[i];
      std::list<std::string> new_strings = compute_flight_string(*connection);

      if (connection->code == airport.code) {
        #pragma omp critical
        local_flight_strings.merge(new_strings);
      } else {
        for (const std::string &string: new_strings) {
          std::string flight_string =
            airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
          #pragma omp critical
          local_flight_strings.emplace_back(flight_string + ", " + string);
        }
      }
    }

    #pragma omp critical
    flight_strings.merge(local_flight_strings);
  }

  return flight_strings;
}


int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cerr << "Input file missing." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Input file name
  std::string input_filename = argv[1];
  std::set<int> timesteps;
  std::map<std::string, Airport> start_airports;

  // Read input data
  std::vector<Flight> flights = read_input_file(input_filename, timesteps, start_airports);

  // Build the time-expanded flight graph (equigraph)
  auto timestep_airports = compute_equigraph(flights, timesteps, start_airports);

  // Validate the graph for feasibility
  for (auto it = timesteps.begin(); it != timesteps.end(); ++it) {
    for (auto &pair: timestep_airports[*it]) {
      Airport &airport = pair.second;
      if (airport.num_planes < airport.num_departures) {
        std::cout << "Flight routing is infeasible with given flights." << std::endl;
        exit(EXIT_SUCCESS);
      }
    }
  }

  // Prepare a vector of iterators for OpenMP parallelization
  std::vector<std::map<std::string, Airport>::iterator> airport_iterators;
  for (auto it = start_airports.begin(); it != start_airports.end(); ++it) {
    airport_iterators.push_back(it);
  }

  // Parallel compute flight strings from starting airports
  #pragma omp parallel for
  for (size_t i = 0; i < airport_iterators.size(); ++i) {
    auto it = airport_iterators[i];
    Airport &airport = it->second;
    std::list<std::string> flight_strings = compute_flight_string(airport);

    // Thread-safe output of flight strings
    #pragma omp critical
    {
      for (const std::string &flight_string: flight_strings) {
        std::cout << flight_string << std::endl;
      }
    }
  }

  return 0;
}
