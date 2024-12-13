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

std::vector<std::vector<std::string>> read_routes_file(std::string &input_filename);
std::vector<Flight> read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport*> &airports);
std::map<int, std::map<std::string, Airport*>> compute_equigraph(std::vector<Flight> &flights, std::set<int> &timesteps, std::map<std::string, Airport*> &start_airports,
    int pid, int nproc, int batch_size);
std::list<std::string> compute_flight_string(Airport &airport);
bool compareAirport(Airport* &a, Airport* &b);
void propagate_updates(std::map<int, std::map<std::string, Airport*>> &timestep_airports, std::vector<Airport*> &adjacent_pairs,
    int batch_size, int pairs_to_send, int pid, int nproc, int src, int dst, int* int_send_buf, char* char_send_buf,
    std::vector<int*> &int_recv_bufs, std::vector<char*> &char_recv_bufs);

#endif
