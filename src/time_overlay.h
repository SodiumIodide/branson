#ifndef time_overlay_h_
#define time_overlay_h_

#include <map>

#include "random_overlay.h"
#include "input.h"
#include "imc_state.h"

class Time_Overlay {
public:
  //! Constructor
  Time_Overlay(Input &input) : num_cells(input.get_structured_cells()), problem_dist(input.get_problem_dist()), num_materials(input.get_num_materials()) { }

  void set(Mesh &mesh, const uint32_t &step, const int &rank, const int &n_rank) {
    if (overlays.find(step) == overlays.end()) {
      overlays.insert(std::make_pair(step, new Random_Overlay(num_cells, problem_dist, num_materials)));
    }

    overlays[step]->process_output(mesh, rank, n_rank);
  }

  void print_all(const int &rank) {
    for (std::map<uint32_t, Random_Overlay*>::iterator itr = overlays.begin(); itr != overlays.end(); itr++) {
      itr->second->write_hdf5(rank, itr->first);
    }
  }

  //! Destructor
  ~Time_Overlay() {
    for (std::map<uint32_t, Random_Overlay*>::iterator itr = overlays.begin(); itr != overlays.end(); itr++) {
      delete (itr->second);
    }
    overlays.clear();
  }

private:
  std::map<uint32_t, Random_Overlay*> overlays;
  int num_cells;
  double problem_dist;
  int num_materials;
};

#endif
