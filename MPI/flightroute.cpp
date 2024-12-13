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
      std::cout << "Flight routing is infeasible with given flights." << std::endl;
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
  return a->code < b->code;
}

void propagate_updates(std::map<int, std::map<std::string, Airport*>> &timestep_airports, std::vector<Airport*> &adjacent_pairs,
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

std::map<int, std::map<std::string, Airport*>> compute_equigraph(std::vector<Flight> &flights, std::set<int> &timesteps, std::map<std::string, Airport*> &start_airports,
    int pid, int nproc, int batch_size) {
  std::map<int, std::map<std::string, Airport*>> timestep_airports;
  timestep_airports[-1] = start_airports;

  std::vector<Airport*> timestep_airport_vector;
  for (auto &pair : start_airports) {
    Airport &airport = *(pair.second);
    timestep_airport_vector.push_back(&airport);
  }
  
  // Add edges for flights
  /*
   * Note: Not worth parallelizing with MPI? Propagated updates from other processes would contain info on new Airport nodes; we'd have to
   * initialize local Airport nodes, check if they already exist in the map, and add edges anyway. All of that data is already contained in the Flights.
   */ 
  for (Flight &flight : flights) {
    int depart_timestep = flight.depart_day * 24 + flight.depart_time;
    int arrive_timestep = flight.arrive_day * 24 + flight.arrive_time;
    
    // Initialize departure maps if non-existent
    if (timestep_airports.find(depart_timestep) == timestep_airports.end()) {
      std::map<std::string, Airport*> airports;
      timestep_airports[depart_timestep] = airports;
    }
    std::map<std::string, Airport*> &depart_airports = timestep_airports[depart_timestep];
    if (depart_airports.find(flight.depart_airport) == depart_airports.end()) {
      std::list<Airport*> adj_list;
      depart_airports[flight.depart_airport] = new Airport{0, 0, depart_timestep, flight.depart_airport, adj_list};
      timestep_airport_vector.push_back(depart_airports[flight.depart_airport]);
    }
    depart_airports[flight.depart_airport]->num_departures++;

    // Initialize arrival maps if non-existent
    if (timestep_airports.find(arrive_timestep) == timestep_airports.end()) {
      std::map<std::string, Airport*> airports;
      timestep_airports[arrive_timestep] = airports;
    }
    std::map<std::string, Airport*> &arrive_airports = timestep_airports[arrive_timestep];
    if (arrive_airports.find(flight.arrive_airport) == arrive_airports.end()) {
      std::list<Airport*> adj_list;
      arrive_airports[flight.arrive_airport] = new Airport{0, 0, arrive_timestep, flight.arrive_airport, adj_list};
      timestep_airport_vector.push_back(arrive_airports[flight.arrive_airport]);
    }
    arrive_airports[flight.arrive_airport]->num_planes++;

    // Add edge to graph
    timestep_airports[depart_timestep][flight.depart_airport]->adj_list.push_back(timestep_airports[arrive_timestep][flight.arrive_airport]);
  }

  int src = (pid == 0) ? nproc - 1 : pid - 1;
  int dst = (pid == nproc - 1) ? 0 : pid + 1;

  int* int_send_buf = new int[batch_size * 6 + 2];
  char* char_send_buf = new char[batch_size * 3];
  std::vector<int*> int_recv_bufs(nproc);
  std::vector<char*> char_recv_bufs(nproc);
  for (int i = 0; i < nproc; i++) int_recv_bufs[i] = new int[batch_size * 6 + 2];
  for (int i = 0; i < nproc; i++) char_recv_bufs[i] = new char[batch_size * 3];

  // Add edges for ground connections
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
    propagate_updates(timestep_airports, adjacent_pairs, batch_size, pairs_to_send, pid, nproc, src, dst, int_send_buf, char_send_buf,
      int_recv_bufs, char_recv_bufs);
  }

  // If any processors finish all their batches while other processors are still working, need to continue propagating updates (otherwise the ring breaks)
  int remaining_nodes = timestep_airport_vector.size() % (nproc * batch_size);
  int active_procs = remaining_nodes / batch_size;
  bool batches_full = (remaining_nodes % batch_size == 0);
  if ((batches_full && pid >= active_procs) || (!batches_full && pid > active_procs)) {
    // Note: Use batch size 0 since this processor is done with batches, and -pid as offset for unique identifier
    std::vector<Airport*> empty;
    propagate_updates(timestep_airports, empty, batch_size, 0, -pid, nproc, src, dst, int_send_buf, char_send_buf, int_recv_bufs, char_recv_bufs);
  }

  // Clean up
  delete[] int_send_buf;
  delete[] char_send_buf;
  for (int i = 0; i < nproc; i++) {
    delete[] int_recv_bufs[i];
    delete[] char_recv_bufs[i];
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
  if (argc <= 2) {
    std::cerr << "Inputs missing." << std::endl;
    exit(EXIT_FAILURE);
  }

  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const auto init_start = std::chrono::steady_clock::now();

  //read_routes_file(ROUTES_FILE);
  std::string input_filename = argv[1];
  int batch_size = std::stoi(argv[2]);
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
    std::cout << "Batch size: " << batch_size << std::endl;

    flights = read_input_file(input_filename, timesteps, start_airports);

    // Note: Each airport code is exactly 3 letters
    num_flights = flights.size();
    num_timesteps = timesteps.size();
    num_start_airports = start_airports.size();
    flight_buf = new int[num_flights * 4];
    flight_char_buf = new char[num_flights * 6];
    timestep_buf = new int[num_timesteps];
    start_airport_buf = new int[num_start_airports * 3];
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
      start_airport_buf[sa++] = airport.num_departures;
      start_airport_buf[sa++] = airport.timestep;
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
    start_airport_buf = new int[num_start_airports * 3];
    start_airport_char_buf = new char[num_start_airports * 3];
  }

  MPI_Bcast(flight_buf, num_flights * 4, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(flight_char_buf, num_flights * 6, MPI_CHAR, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(timestep_buf, num_timesteps, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(start_airport_buf, num_start_airports * 3, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(start_airport_char_buf, num_start_airports * 3, MPI_CHAR, ROOT, MPI_COMM_WORLD);

  if (pid != ROOT) {
    for (int i = 0; i < num_timesteps; i++)
      timesteps.insert(timestep_buf[i]);
    
    for (int i = 0; i < num_start_airports * 3; i += 3) {
      std::string code{start_airport_char_buf[i], start_airport_char_buf[i+1], start_airport_char_buf[i+2]};
      int num_planes = start_airport_buf[i];
      int num_departures = start_airport_buf[i+1];
      int timestep = start_airport_buf[i+2];
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

  //std::vector<int> timesteps(timestep_set.begin(), timestep_set.end());
  std::map<int, std::map<std::string, Airport*>> timestep_airports = compute_equigraph(flights, timesteps, start_airports, pid, nproc, batch_size);

  std::set<int>::iterator it;
  for (it = timesteps.begin(); it != timesteps.end(); it++) {
    for (auto &pair : timestep_airports[*it]) {
      std::string code = pair.first;
      Airport &airport = *(pair.second);
      if (airport.num_planes < airport.num_departures) {
        std::cout << "Flight routing is infeasible with given flights." << std::endl;
        MPI_Finalize();
        exit(EXIT_SUCCESS);
      }
    }
  }

  if (pid == ROOT) { // TODO: COMPLETE PARALLELIZATION FOR DFS
    for (auto &pair : start_airports) {
      Airport &airport = *(pair.second);
      std::list<std::string> flight_strings = compute_flight_string(airport);
      for (std::string flight_string : flight_strings)
        std::cout << flight_string << std::endl;
    }
  }

  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
  }

  MPI_Finalize();
  return 0;
}
