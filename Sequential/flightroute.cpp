/**
 * Parallel Flight Routing (Sequential Version)
 * Jesse Liu (jzliu), Oscar Han (Enxuh)
 */

//#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

//#include <unistd.h>
//#include <stdlib.h>
//#include <cmath>
//#include <climits>
//#include <utility>

#include "flightroute.h"

std::vector<std::vector<std::string>> read_routes_file(std::string &input_filename) {
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  std::vector<std::vector<std::string>> route_data;

  // Skip first line
  std::string line;
  std::getline(fin, line);

  while (std::getline(fin, line)) {
    std::vector<std::string> fields;
    std::istringstream iss(line);
    std::string field;

    while (std::getline(iss, field, ',')) {
      std::cout << field << std::endl;
      fields.push_back(field);
    }

    if (!iss && field.empty()) { // Check for empty equipment field
      fields.push_back("");
    }
    std::cout << std::endl;

    route_data.push_back(fields);
  }

  return route_data;
}

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

std::list<std::string> compute_flight_string(Airport &airport) {
  std::list<std::string> flight_strings;

  if (airport.adj_list.size() == 0) {
    std::string flight_string = airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
    flight_strings.emplace_back(flight_string);
    return flight_strings;
  }

  for (Airport* connection : airport.adj_list) {
    std::list<std::string> new_strings = compute_flight_string(*connection);
    if (connection->code == airport.code) {
      flight_strings.merge(new_strings);
    } else {
      for (std::string string : new_strings) {
        std::string flight_string = airport.code + ':' + std::to_string(airport.timestep / 24) + ':' + std::to_string(airport.timestep % 24);
        flight_strings.emplace_back(flight_string + ", " + string);
      }
    }
  }

  return flight_strings;
}

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cerr << "Input file missing." << std::endl;
    exit(EXIT_FAILURE);
  }

  const auto init_start = std::chrono::steady_clock::now();

  //read_routes_file(ROUTES_FILE);
  std::string input_filename = argv[1];  
  std::set<int> timesteps;
  std::map<std::string, Airport> start_airports;

  std::vector<Flight> flights = read_input_file(input_filename, timesteps, start_airports);
  //std::vector<int> timesteps(timestep_set.begin(), timestep_set.end());
  std::map<int, std::map<std::string, Airport>> timestep_airports = compute_equigraph(flights, timesteps, start_airports);

  std::set<int>::iterator it;
  for (it = timesteps.begin(); it != timesteps.end(); it++) {
    for (auto &pair : timestep_airports[*it]) {
      std::string code = pair.first;
      Airport &airport = pair.second;
      if (airport.num_planes < airport.num_departures) {
        std::cout << "Flight routing is infeasible with given flights." << std::endl;
        exit(EXIT_SUCCESS);
      }
    }
  }

  for (auto &pair : start_airports) {
    Airport &airport = pair.second;
    std::list<std::string> flight_strings = compute_flight_string(airport);
    for (std::string flight_string : flight_strings)
      std::cout << flight_string << std::endl;
  }

  const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
  std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
  
  return 0;
}
