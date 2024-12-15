/**
 * Parallel Flight Routing (MPI Version)
 * Jesse Liu (jzliu), Oscar Han (Enxuh)
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>

#include <mpi.h>
//#include <unistd.h>
//#include <stdlib.h>
//#include <cmath>
//#include <climits>
//#include <utility>

#include "flightroute.h"

#define ROOT 0

std::vector<std::vector<std::string>> read_routes_file(std::string &input_filename) {
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    MPI_Finalize();
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

std::vector<Flight> read_input_file(std::string &input_filename, std::set<int> &timesteps, std::map<std::string, Airport*> &airports) {
  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    MPI_Finalize();
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
      std::cout << "Flight routing is infeasible with given flight: " << flight.depart_airport << ':' << flight.depart_day << ':' << flight.depart_time
        << ", " << flight.arrive_airport << ':' << flight.arrive_day << ':' << flight.arrive_time << std::endl;
      MPI_Finalize();
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

    airports[airport] = new Airport{aircraft_count, 0, -1, airport, adj_list};
  }

  return flights;
}

bool compareAirport(Airport* &a, Airport* &b) {
  if (a->timestep == b->timestep)
    return a->code < b->code;
  return a->timestep < b->timestep;
}

/*
void propagate_updates_init(std::map<int, std::map<std::string, Airport*>> &timestep_airports, std::vector<Airport*> &adjacent_pairs,
    int batch_size, int pairs_to_send, int pid, int nproc, int src, int dst, int* int_send_buf, char* char_send_buf,
    std::vector<int*> &int_recv_bufs, std::vector<char*> &char_recv_bufs) {
  MPI_Request int_send_request, char_send_request;
  std::vector<MPI_Request> recv_requests(nproc * 2);

  int_send_buf[0] = pid;
  int_send_buf[1] = pairs_to_send;
  for (int i = 0; i < pairs_to_send; i++) {
    Airport &start = *adjacent_pairs[i * 2];
    Airport &end = *adjacent_pairs[i * 2 + 1];
    int_send_buf[i * 6 + 2] = start.num_planes;
    int_send_buf[i * 6 + 3] = start.num_departures;
    int_send_buf[i * 6 + 4] = start.timestep;
    int_send_buf[i * 6 + 5] = end.num_planes;
    int_send_buf[i * 6 + 6] = end.num_departures;
    int_send_buf[i * 6 + 7] = end.timestep;
    char_send_buf[i * 3] = start.code[0];
    char_send_buf[i * 3 + 1] = start.code[1];
    char_send_buf[i * 3 + 2] = start.code[2];
  }
  MPI_Isend(int_send_buf, batch_size * 6 + 2, MPI_INT, dst, 0, MPI_COMM_WORLD, &int_send_request);
  MPI_Isend(char_send_buf, batch_size * 3, MPI_CHAR, dst, 1, MPI_COMM_WORLD, &char_send_request);

  // Synchronously receive nproc messages from left processor in ring
  int pairs_to_recv;
  int ignore_idx = -1; // One message will be originally from this processor; identify through pid and ignore
  for (int i = 0; i < nproc; i++) {
    MPI_Recv(int_recv_bufs[i], batch_size * 6 + 2, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(char_recv_bufs[i], batch_size * 3, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (int_recv_bufs[i][0] == pid) {
      ignore_idx = i;
      continue;
    }

    // Update timestep_airports with updates from other processors
    pairs_to_recv = int_recv_bufs[i][1];
    for (int j = 0; j < pairs_to_recv; j++) {
      int start_num_planes = int_recv_bufs[i][j * 6 + 2];
      int start_num_departures = int_recv_bufs[i][j * 6 + 3];
      int start_timestep = int_recv_bufs[i][j * 6 + 4];
      int end_num_planes = int_recv_bufs[i][j * 6 + 5];
      int end_num_departures = int_recv_bufs[i][j * 6 + 6];
      int end_timestep = int_recv_bufs[i][j * 6 + 7];
      std::string code{char_recv_bufs[i][j * 3], char_recv_bufs[i][j * 3 + 1], char_recv_bufs[i][j * 3 + 2]};

      Airport &start = *(timestep_airports[start_timestep][code]);
      start.num_planes = start_num_planes;
      start.num_departures = start_num_departures;

      Airport &end = *(timestep_airports[end_timestep][code]);
      end.num_planes = end_num_planes;
      end.num_departures = end_num_departures;

      start.adj_list.push_back(&end);
    }

    // Pass messages along
    MPI_Isend(int_recv_bufs[i], batch_size * 6 + 2, MPI_INT, dst, 0, MPI_COMM_WORLD, &recv_requests[i * 2]);
    MPI_Isend(char_recv_bufs[i], batch_size * 3, MPI_CHAR, dst, 1, MPI_COMM_WORLD, &recv_requests[i * 2 + 1]);
  }

  // Finish asynchronous sends
  MPI_Wait(&int_send_request, MPI_STATUS_IGNORE);
  MPI_Wait(&char_send_request, MPI_STATUS_IGNORE);
  for (int i = 0; i < nproc; i++) {
    if (i != ignore_idx) {
      MPI_Wait(&recv_requests[i * 2], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_requests[i * 2 + 1], MPI_STATUS_IGNORE);
    }
  }
}
*/

/*
* Note: Not worth parallelizing with MPI? Propagated updates from other processes would contain info on new Airport nodes and edges; we'd have to
* initialize local Airport nodes, check if they already exist in the map, and add edges anyway. All of that data is already contained in the Flights
* and maps that we build, so we wouldn't really receive any new information from the other processes.
*/ 
std::map<std::string, std::map<int, Airport*>> compute_equigraph(std::vector<Flight> &flights, std::map<std::string, Airport*> &start_airports) {
  std::map<std::string, std::map<int, Airport*>> timestep_airports;
  for (auto &pair : start_airports) {
    Airport* a = pair.second;
    std::map<int, Airport*> new_timestep_map;
    new_timestep_map[-1] = a;
    timestep_airports[a->code] = new_timestep_map;
  }
  
  // Add edges for flights
  for (Flight &flight : flights) {
    int depart_timestep = flight.depart_day * 24 + flight.depart_time;
    int arrive_timestep = flight.arrive_day * 24 + flight.arrive_time;
    
    // Initialize departure maps if non-existent
    if (timestep_airports.find(flight.depart_airport) == timestep_airports.end()) {
      std::map<int, Airport*> airports;
      timestep_airports[flight.depart_airport] = airports;
    }
    std::map<int, Airport*> &depart_airports = timestep_airports[flight.depart_airport];
    if (depart_airports.find(depart_timestep) == depart_airports.end()) {
      std::list<Airport*> adj_list;
      depart_airports[depart_timestep] = new Airport{0, 0, depart_timestep, flight.depart_airport, adj_list};
    }
    depart_airports[depart_timestep]->num_departures++;

    // Initialize arrival maps if non-existent
    if (timestep_airports.find(flight.arrive_airport) == timestep_airports.end()) {
      std::map<int, Airport*> airports;
      timestep_airports[flight.arrive_airport] = airports;
    }
    std::map<int, Airport*> &arrive_airports = timestep_airports[flight.arrive_airport];
    if (arrive_airports.find(arrive_timestep) == arrive_airports.end()) {
      std::list<Airport*> adj_list;
      arrive_airports[arrive_timestep] = new Airport{0, 0, arrive_timestep, flight.arrive_airport, adj_list};
    }
    arrive_airports[arrive_timestep]->num_planes++;

    // Add edge to graph
    timestep_airports[flight.depart_airport][depart_timestep]->adj_list.push_back(timestep_airports[flight.arrive_airport][arrive_timestep]);
  }

  for (auto &pair : timestep_airports) {
    std::map<int, Airport*> &timestep_map = pair.second;
    auto curr = timestep_map.begin();
    auto prev = curr;
    while (++curr != timestep_map.end()) {
      (prev->second)->adj_list.push_back(curr->second);
      (curr->second)->num_planes += ((prev->second)->num_planes - (prev->second)->num_departures);
      prev++;
    }
  }

  /*
  int src = (pid == 0) ? nproc - 1 : pid - 1;
  int dst = (pid == nproc - 1) ? 0 : pid + 1;

  int* int_send_buf = new int[batch_size * 6 + 2];
  char* char_send_buf = new char[batch_size * 3];
  std::vector<int*> int_recv_bufs(nproc);
  std::vector<char*> char_recv_bufs(nproc);
  for (int i = 0; i < nproc; i++) int_recv_bufs[i] = new int[batch_size * 6 + 2];
  for (int i = 0; i < nproc; i++) char_recv_bufs[i] = new char[batch_size * 3];

  // Add edges for ground connections
  // NOTE: Airport nodes for previous timesteps must be computed before nodes for the same airport at later timesteps; same airport cannot be done in parallel
  std::sort(timestep_airport_vector.begin(), timestep_airport_vector.end(), compareAirport);
  for (int i = pid * batch_size; i < timestep_airport_vector.size(); i += nproc * batch_size) {
    int actual_batch_size = std::min(batch_size, (int) timestep_airport_vector.size() - i);
    std::vector<Airport*> adjacent_pairs;

    for (int j = 0; j < actual_batch_size; j++) {
      Airport &airport = *timestep_airport_vector[i + j];
      int curr_timestep = airport.timestep;
      std::set<int>::iterator it;
      if (curr_timestep == -1) it = timesteps.begin();
      else it = ++timesteps.find(curr_timestep); // Start at next timestep

      while (it != timesteps.end() && timestep_airports[*it].find(airport.code) == timestep_airports[*it].end())
        it++;
      
      if (it != timesteps.end()) {
        std::map<std::string, Airport*> &airports = timestep_airports[*it];
        airport.adj_list.push_back(airports[airport.code]);
        airports[airport.code]->num_planes += (airport.num_planes - airport.num_departures);

        adjacent_pairs.push_back(&airport);
        adjacent_pairs.push_back(airports[airport.code]);
      }
    }
    
    int pairs_to_send = adjacent_pairs.size() / 2;
    // Note: Processors arranged in ring; send to right, receive from left, pass messages along after receiving
    propagate_updates_init(timestep_airports, adjacent_pairs, batch_size, pairs_to_send, pid, nproc, src, dst, int_send_buf, char_send_buf,
      int_recv_bufs, char_recv_bufs);
  }

  // If any processors finish all their batches while other processors are still working, need to continue propagating updates (otherwise the ring breaks)
  int remaining_nodes = timestep_airport_vector.size() % (nproc * batch_size);
  int active_procs = remaining_nodes / batch_size;
  bool batches_full = (remaining_nodes % batch_size == 0);
  if ((batches_full && pid >= active_procs) || (!batches_full && pid > active_procs)) {
    // Note: Use batch size 0 since this processor is done with batches, and -pid as offset for unique identifier
    std::vector<Airport*> empty;
    propagate_updates_init(timestep_airports, empty, batch_size, 0, -pid, nproc, src, dst, int_send_buf, char_send_buf, int_recv_bufs, char_recv_bufs);
  }

  // Clean up
  delete[] int_send_buf;
  delete[] char_send_buf;
  for (int i = 0; i < nproc; i++) {
    delete[] int_recv_bufs[i];
    delete[] char_recv_bufs[i];
  }
  */

  return timestep_airports;
}

/*
Get adj list len for start airports (+1 because they all start at -1 and have len 1)
Batch inputs for each start airport? Or just have diff procs run on diff start airports? What if imbalanced adj list lens?
Compute flight strings, first round propagate max string len and num strings
Each proc gets overall max string len and max num strings
Second round propagate strings, each proc adds to their total list
Repeat until done
Potentially terrible work balance/distribution if graph is too deep, width is fine, but no way to know how many departures you have
on connecting flights on the nodes deeper in the graph at the start
*/
void propagate_updates_first(int src, int dst, int pid, int nproc, std::size_t* max_flight_string_len, std::size_t* num_flight_strings,
    int* send_buf, std::vector<int*> &recv_bufs) {
  MPI_Request send_request;
  std::vector<MPI_Request> recv_requests(nproc);

  send_buf[0] = pid;
  send_buf[1] = *max_flight_string_len;
  send_buf[2] = *num_flight_strings;
  MPI_Isend(send_buf, 3, MPI_INT, dst, 0, MPI_COMM_WORLD, &send_request);

  // Synchronously receive nproc messages from left processor in ring
  int ignore_idx = -1; // One message will be originally from this processor; identify through pid and ignore
  for (int i = 0; i < nproc; i++) {
    MPI_Recv(recv_bufs[i], 3, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (recv_bufs[i][0] == pid) {
      ignore_idx = i;
      continue;
    }

    // Update max length and number of flight strings with updates from other processors
    std::size_t flight_string_len = recv_bufs[i][1];
    std::size_t num_flight_string = recv_bufs[i][2];
    *max_flight_string_len = std::max(*max_flight_string_len, flight_string_len);
    *num_flight_strings = std::max(*num_flight_strings, num_flight_string);

    // Pass messages along
    MPI_Isend(recv_bufs[i], 3, MPI_INT, dst, 0, MPI_COMM_WORLD, &recv_requests[i]);
  }

  // Finish asynchronous sends
  MPI_Wait(&send_request, MPI_STATUS_IGNORE);
  for (int i = 0; i < nproc; i++) {
    if (i != ignore_idx) {
      MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
    }
  }
}

// TODO: Send strings from curr_flight_strings to other procs, recved strings add to new_flight_strings
void propagate_updates_second(int src, int dst, int pid, int nproc, std::size_t max_flight_string_len, std::size_t num_flight_strings,
    std::list<std::string> &new_flight_strings, std::list<std::string> &curr_flight_strings, int* int_send_buf, std::vector<int*> &int_recv_bufs,
    char* char_send_buf, std::vector<char*> &char_recv_bufs) {
  MPI_Request int_send_request;
  MPI_Request char_send_request;
  std::vector<MPI_Request> int_recv_requests(nproc);
  std::vector<MPI_Request> char_recv_requests(nproc);

  int char_idx = 0;
  int_send_buf[0] = pid;
  int_send_buf[1] = (int) curr_flight_strings.size();
  auto it = curr_flight_strings.begin();
  for (int i = 0; i < int_send_buf[1]; i++) {
    std::string &str = *(it++);
    int_send_buf[i + 2] = (int) str.size();
    for (int j = 0; j < int_send_buf[i + 2]; j++) char_send_buf[char_idx++] = str[j];
  }
  MPI_Isend(int_send_buf, num_flight_strings + 2, MPI_INT, dst, 0, MPI_COMM_WORLD, &int_send_request);
  MPI_Isend(char_send_buf, max_flight_string_len * num_flight_strings, MPI_CHAR, dst, 1, MPI_COMM_WORLD, &char_send_request);

  // Synchronously receive nproc messages from left processor in ring
  int ignore_idx = -1; // One message will be originally from this processor; identify through pid and ignore
  for (int i = 0; i < nproc; i++) {
    MPI_Recv(int_recv_bufs[i], num_flight_strings + 2, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(char_recv_bufs[i], max_flight_string_len * num_flight_strings, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (int_recv_bufs[i][0] == pid) {
      ignore_idx = i;
      continue;
    }

    // Update new_flight_strings with updates from other processors
    char_idx = 0;
    int num_new_strings = int_recv_bufs[i][1];
    for (int j = 0; j < num_new_strings; j++) {
      int new_string_len = int_recv_bufs[i][j + 2];
      std::string new_string = "";
      for (int k = 0; k < new_string_len; k++) new_string += char_recv_bufs[i][char_idx++];
      new_flight_strings.push_back(new_string);
    }

    // Pass messages along
    MPI_Isend(int_recv_bufs[i], num_flight_strings + 2, MPI_INT, dst, 0, MPI_COMM_WORLD, &int_recv_requests[i]);
    MPI_Isend(char_recv_bufs[i], max_flight_string_len * num_flight_strings, MPI_CHAR, dst, 1, MPI_COMM_WORLD, &char_recv_requests[i]);
  }

  // Finish asynchronous sends
  MPI_Wait(&int_send_request, MPI_STATUS_IGNORE);
  MPI_Wait(&char_send_request, MPI_STATUS_IGNORE);
  for (int i = 0; i < nproc; i++) {
    if (i != ignore_idx) {
      MPI_Wait(&int_recv_requests[i], MPI_STATUS_IGNORE);
      MPI_Wait(&char_recv_requests[i], MPI_STATUS_IGNORE);
    }
  }
}

// Compute flight strings for a single airport
std::list<std::string> compute_flight_string(Airport* &airport, std::unordered_set<Airport*> &visited, std::size_t* max_flight_string_len) {
  std::list<std::string> flight_strings;

  // Check if the airport has already been visited
  if (visited.find(airport) != visited.end()) {
    std::string flight_string =
      airport->code + ':' + std::to_string(airport->timestep / 24) + ':' + std::to_string(airport->timestep % 24);
    flight_strings.emplace_back(flight_string);
    *max_flight_string_len = std::max(*max_flight_string_len, flight_string.length());
    return flight_strings; // Return an empty list to prevent cycles
  }

  // Mark the current airport as visited
  visited.insert(airport);

  // If no adjacent airports, this is a terminal node
  if (airport->adj_list.empty()) {
    std::string flight_string =
      airport->code + ':' + std::to_string(airport->timestep / 24) + ':' + std::to_string(airport->timestep % 24);
    *max_flight_string_len = std::max(*max_flight_string_len, flight_string.length());
    flight_strings.emplace_back(flight_string);
  } else {
    // Recursively compute flight strings for each connection
    for (Airport* &connection: airport->adj_list) {
      std::list<std::string> new_strings = compute_flight_string(connection, visited, max_flight_string_len);
      for (const std::string &string: new_strings) {
        std::string flight_string =
          airport->code + ':' + std::to_string(airport->timestep / 24) + ':' + std::to_string(airport->timestep % 24) +
          " -> " + string;
        *max_flight_string_len = std::max(*max_flight_string_len, flight_string.length());
        flight_strings.emplace_back(flight_string);
      }
    }
  }

  return flight_strings;
}

// Compute flight strings for all airports
std::list<std::list<std::string>> compute_flight_strings(std::map<std::string, Airport*> &airports, int pid, int nproc) {
  std::list<std::list<std::string>> all_flight_strings;
  int src = (pid == 0) ? nproc - 1 : pid - 1;
  int dst = (pid == nproc - 1) ? 0 : pid + 1;

  // Add starting airports to a vector, in order to divide among processors
  std::vector<Airport*> first_timestep_airports;
  for (auto &pair : airports) {
    Airport* &airport = pair.second->adj_list.front(); // Note: Starting airport has timestep -1, adjacency list always goes solely to copy at next relevant timestep
    first_timestep_airports.push_back(airport);
  }
  
  // Propagate stage 1 (max flight string length and num flight strings)
  // Note: Starting airports divided by nproc, no benefits from having more processors than starting airports
  // Note 2: Workload distribution may be imbalanced, difficult to judge due to unknown depth
  std::size_t max_flight_string_len = 0;
  std::size_t num_flight_strings = 0;
  int* send_buf = new int[3];
  std::vector<int*> recv_bufs(nproc);
  for (int i = 0; i < nproc; i++) recv_bufs[i] = new int[3];

  for (std::size_t i = pid; i < first_timestep_airports.size(); i += nproc) {
    Airport* &airport = first_timestep_airports[i];
    auto visited = std::unordered_set<Airport*>();
    
    std::list<std::string> airport_flight_strings = compute_flight_string(airport, visited, &max_flight_string_len);
    num_flight_strings = std::max(num_flight_strings, airport_flight_strings.size());

    propagate_updates_first(src, dst, pid, nproc, &max_flight_string_len, &num_flight_strings, send_buf, recv_bufs);

    // Add the flight strings for this airport as a separate list
    all_flight_strings.push_back(std::move(airport_flight_strings));
  }

  // If any processors finish all their batches while other processors are still working, need to continue propagating updates (otherwise the ring breaks)
  int remaining_nodes = first_timestep_airports.size() % nproc;
  if (pid >= remaining_nodes) {
    propagate_updates_first(src, dst, -pid, nproc, &max_flight_string_len, &num_flight_strings, send_buf, recv_bufs);
  }

  // Clean up stage 1
  delete[] send_buf;
  for (int i = 0; i < nproc; i++) delete[] recv_bufs[i];

  // Propagate stage 2 (actual flight strings)
  // Note: Potentially huge memory overhead?
  std::list<std::string> new_flight_strings;
  int* int_send_buf = new int[num_flight_strings + 2];
  char* char_send_buf = new char[max_flight_string_len * num_flight_strings];
  std::vector<int*> int_recv_bufs(nproc);
  for (int i = 0; i < nproc; i++) int_recv_bufs[i] = new int[num_flight_strings + 2];
  std::vector<char*> char_recv_bufs(nproc);
  for (int i = 0; i < nproc; i++) char_recv_bufs[i] = new char[max_flight_string_len * num_flight_strings];

  for (std::list<std::string> &airport_flight_strings : all_flight_strings) {
    // Currently sends whole list of strings in one msg, can split into multiple if it doesn't fit, but worse communication overhead?
    propagate_updates_second(src, dst, pid, nproc, max_flight_string_len, num_flight_strings, new_flight_strings, airport_flight_strings,
      int_send_buf, int_recv_bufs, char_send_buf, char_recv_bufs);
  }

  // If any processors finish all their batches while other processors are still working, need to continue propagating updates (otherwise the ring breaks)
  if (pid >= remaining_nodes) {
    std::list<std::string> empty;
    propagate_updates_second(src, dst, -pid, nproc, max_flight_string_len, num_flight_strings, new_flight_strings, empty,
      int_send_buf, int_recv_bufs, char_send_buf, char_recv_bufs);
  }

  all_flight_strings.push_back(std::move(new_flight_strings));

  // Clean up stage 2
  delete[] int_send_buf;
  delete[] char_send_buf;
  for (int i = 0; i < nproc; i++) delete[] int_recv_bufs[i];
  for (int i = 0; i < nproc; i++) delete[] char_recv_bufs[i];

  return all_flight_strings;
}

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cerr << "Inputs missing." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize timers
  const auto init_start = std::chrono::high_resolution_clock::now();

  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  //read_routes_file(ROUTES_FILE);
  std::string input_filename = argv[1];
  int num_flights;
  int num_timesteps;
  int num_start_airports;
  int* flight_buf;
  char* flight_char_buf;
  int* timestep_buf;
  int* start_airport_buf;
  char* start_airport_char_buf;
  std::set<int> timesteps;
  std::map<std::string, Airport*> start_airports;
  std::vector<Flight> flights;

  if (pid == ROOT) {
    std::cout << "Number of processes: " << nproc << std::endl;
    std::cout << "Input file: " << input_filename << std::endl;

    flights = read_input_file(input_filename, timesteps, start_airports);

    // Note: Each airport code is exactly 3 letters
    num_flights = flights.size();
    num_timesteps = timesteps.size();
    num_start_airports = start_airports.size();
    flight_buf = new int[num_flights * 4];
    flight_char_buf = new char[num_flights * 6];
    timestep_buf = new int[num_timesteps];
    start_airport_buf = new int[num_start_airports];
    start_airport_char_buf = new char[num_start_airports * 3];

    int f = 0;
    int fc = 0;
    for (Flight flight : flights) {
      flight_buf[f++] = flight.depart_day;
      flight_buf[f++] = flight.depart_time;
      flight_buf[f++] = flight.arrive_day;
      flight_buf[f++] = flight.arrive_time;
      flight_char_buf[fc++] = flight.depart_airport[0];
      flight_char_buf[fc++] = flight.depart_airport[1];
      flight_char_buf[fc++] = flight.depart_airport[2];
      flight_char_buf[fc++] = flight.arrive_airport[0];
      flight_char_buf[fc++] = flight.arrive_airport[1];
      flight_char_buf[fc++] = flight.arrive_airport[2];
    }

    int t = 0;
    for (int timestep : timesteps) {
      timestep_buf[t++] = timestep;
    }

    int sa = 0;
    int sac = 0;
    for (auto &pair : start_airports) {
      Airport &airport = *(pair.second);
      start_airport_buf[sa++] = airport.num_planes;
      start_airport_char_buf[sac++] = airport.code[0];
      start_airport_char_buf[sac++] = airport.code[1];
      start_airport_char_buf[sac++] = airport.code[2];
    }
  }

  MPI_Bcast(&num_flights, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&num_timesteps, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&num_start_airports, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

  if (pid != ROOT) {
    flight_buf = new int[num_flights * 4];
    flight_char_buf = new char[num_flights * 6];
    timestep_buf = new int[num_timesteps];
    start_airport_buf = new int[num_start_airports];
    start_airport_char_buf = new char[num_start_airports * 3];
  }

  MPI_Bcast(flight_buf, num_flights * 4, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(flight_char_buf, num_flights * 6, MPI_CHAR, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(timestep_buf, num_timesteps, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(start_airport_buf, num_start_airports, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(start_airport_char_buf, num_start_airports * 3, MPI_CHAR, ROOT, MPI_COMM_WORLD);

  if (pid != ROOT) {
    for (int i = 0; i < num_timesteps; i++)
      timesteps.insert(timestep_buf[i]);
    
    for (int i = 0; i < num_start_airports; i++) {
      std::string code{start_airport_char_buf[i*3], start_airport_char_buf[i*3+1], start_airport_char_buf[i*3+2]};
      int num_planes = start_airport_buf[i];
      int num_departures = 0;
      int timestep = -1;
      std::list<Airport*> adj_list;
      start_airports[code] = new Airport{num_planes, num_departures, timestep, code, adj_list};
    }
    
    for (int i = 0; i < num_flights; i++) {
      int depart_day = flight_buf[i*4];
      int depart_time = flight_buf[i*4+1];
      int arrive_day = flight_buf[i*4+2];
      int arrive_time = flight_buf[i*4+3];
      std::string depart_airport{flight_char_buf[i*6], flight_char_buf[i*6+1], flight_char_buf[i*6+2]};
      std::string arrive_airport{flight_char_buf[i*6+3], flight_char_buf[i*6+4], flight_char_buf[i*6+5]};

      Flight flight = {depart_day, depart_time, arrive_day, arrive_time, depart_airport, arrive_airport};
      flights.push_back(flight);
    }
  }

  delete[] flight_buf;
  delete[] flight_char_buf;
  delete[] timestep_buf;
  delete[] start_airport_buf;
  delete[] start_airport_char_buf;

  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - init_start).count();
    std::cout << "Initialization time: " << std::fixed << std::setprecision(10) << compute_time << std::endl;
  }

  const auto graph_start = std::chrono::high_resolution_clock::now();
  std::map<std::string, std::map<int, Airport*>> timestep_airports = compute_equigraph(flights, start_airports);
  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - graph_start).count();
    std::cout << "Graph construction time: " << std::fixed << std::setprecision(10) << compute_time << std::endl;
  }

  const auto valid_start = std::chrono::high_resolution_clock::now();
  for (auto &pair : timestep_airports) {
    std::string code = pair.first;
    std::map<int, Airport*> timestep_map = pair.second;
    for (auto &pair2 : timestep_map) {
      Airport* airport = pair2.second;
      if (airport->num_planes < airport->num_departures) {
        std::cout << "timestamp: " << airport->timestep << " airport: " << airport->code << " planes: " << airport->num_planes
                  << " departures: " << airport->num_departures << std::endl;
        std::cout << "Flight routing is infeasible with given flights." << std::endl;
        MPI_Finalize();
        exit(EXIT_SUCCESS);
      }
    }
  }
  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - valid_start).count();
    std::cout << "Validation time: " << std::fixed << std::setprecision(10) << compute_time << std::endl;
  }

  // Compute the flight strings
  const auto compute_start = std::chrono::high_resolution_clock::now();
  std::list<std::list<std::string>> flight_strings = compute_flight_strings(start_airports, pid, nproc);
  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - compute_start).count();
    std::cout << "Flight strings computation time: " << std::fixed << std::setprecision(10) << compute_time << std::endl;

    const double total_compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - init_start).count();
    std::cout << "Total computation time: " << std::fixed << std::setprecision(10) << total_compute_time << '\n';
  }

  MPI_Finalize(); // Sometimes hangs on this call?
  return EXIT_SUCCESS;
}
