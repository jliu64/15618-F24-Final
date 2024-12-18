#ifndef __FLIGHTROUTE_H__
#define __FLIGHTROUTE_H__

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
    int num_planes, num_departures, timestep;
    std::string code;
    std::list<Airport*> adj_list;
};

std::vector<Flight> read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport> &airports);
std::map<int, std::map<std::string, Airport>> compute_equigraph(std::vector<Flight> &flights, std::set<int> &timesteps, std::map<std::string, Airport> &start_airports);

#endif
