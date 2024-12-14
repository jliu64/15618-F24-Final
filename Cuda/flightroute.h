#ifndef __FLIGHTROUTE_H__
#define __FLIGHTROUTE_H__

#define MAX_CONNECTIONS 64

#include <string>
#include <vector>
#include <set>
#include <map>
#include <list>

const std::string ROUTES_FILE = "../Data/routes_small.csv";

struct Flight {
  int depart_day, depart_time, arrive_day, arrive_time;
  std::string depart_airport, arrive_airport;
};

struct Airport {
  char code[4]; // Replace std::string with fixed-size char array for CUDA compatibility
  int timestep;
  int adj_list_size; // Number of connections
  Airport *adj_list[MAX_CONNECTIONS]; // Fixed-size array for device compatibility
};

std::vector<std::vector<std::string>> read_routes_file(std::string &input_filename);

std::vector<Flight>
read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport> &airports);

std::map<int, std::map<std::string, Airport>> compute_equigraph(std::vector<Flight> &flights, std::set<int> &timesteps,
                                                                std::map<std::string, Airport> &start_airports);


#endif
