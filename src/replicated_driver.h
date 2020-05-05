//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   replicated_driver.h
 * \author Alex Long
 * \date   March 3 2017
 * \brief  Functions to run IMC with a replicated domain
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef replicated_driver_h_
#define replicated_driver_h_

#include <functional>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "census_creation.h"
#include "imc_parameters.h"
#include "imc_state.h"
#include "info.h"
#include "mesh.h"
#include "mpi_types.h"
#include "replicated_transport.h"
#include "timer.h"
#include "write_silo.h"
#include "write_hdf5.h"
#include "time_overlay.h"

void imc_replicated_driver(Mesh &mesh, IMC_State &imc_state,
                           const IMC_Parameters &imc_parameters,
                           const MPI_Types &mpi_types, const Info &mpi_info, const bool write_output, const bool random_problem, Time_Overlay &overlays) {
  using std::vector;
  vector<double> abs_E(mesh.get_global_num_cells(), 0.0);
  vector<double> track_E(mesh.get_global_num_cells(), 0.0);
  vector<Photon> census_photons;
  int rank = mpi_info.get_rank();
  constexpr double fake_mpi_runtime = 0.0;

  // set the max census size to be 10% more than the user requested photons.
  // Note that this does not need to be divided by the number of ranks
  // because the comb get the global census energy with an MPI_Allreduce
  uint64_t max_census_size =
      static_cast<uint64_t>(1.1 * imc_parameters.get_n_user_photon());

  /*
  if (imc_parameters.get_write_silo_flag() && write_output) {
    // write SILO file
    write_silo(mesh, imc_state.get_time(), imc_state.get_step(),
               imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank,
               mpi_info.get_n_rank());
    // else write HDF5 file
    write_hdf5(mesh, imc_state.get_time(), imc_state.get_step(), imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank, mpi_info.get_n_rank());
  }
  */

  while (!imc_state.finished()) {
    if (write_output && rank == 0)
      imc_state.print_timestep_header();

    // set opacity, Fleck factor, all energy to source
    mesh.calculate_photon_energy(imc_state);

    // all reduce to get total source energy to make correct number of
    // particles on each rank
    double global_source_energy = mesh.get_total_photon_E();
    MPI_Allreduce(MPI_IN_PLACE, &global_source_energy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    imc_state.set_pre_census_E(get_photon_list_E(census_photons));

    // setup source
    Source source(mesh, imc_state, imc_parameters.get_n_user_photon(),
                  global_source_energy, census_photons);
    // no load balancing in particle passing method, just call the method
    // to get accurate count and map census to work correctly
    source.post_lb_prepare_source();

    imc_state.set_transported_particles(source.get_n_photon());

    // transport photons, return the particles that reached census
    census_photons = replicated_transport(source, mesh, imc_state, abs_E,
                                          track_E, max_census_size);

    // reduce the abs_E and the track weighted energy (for T_r)
    MPI_Allreduce(MPI_IN_PLACE, &abs_E[0], mesh.get_global_num_cells(),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &track_E[0], mesh.get_global_num_cells(),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    mesh.update_temperature(abs_E, track_E, imc_state);

    // for replicated let root do conservation checks, zero out your tallies
    if (rank) {
      imc_state.set_absorbed_E(0.0);
      imc_state.set_pre_mat_E(0.0);
      imc_state.set_post_mat_E(0.0);
    }

    if (write_output) {
      imc_state.print_conservation();

      if (imc_parameters.get_write_silo_flag() && !(imc_state.get_step() % imc_parameters.get_output_frequency())) {
        // write SILO file
        write_silo(mesh, imc_state.get_time(), imc_state.get_step(),
                   imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank,
                   mpi_info.get_n_rank());
        // else write HDF5 file
        write_hdf5(mesh, imc_state.get_time(), imc_state.get_step(), imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank, mpi_info.get_n_rank());
      }
    }

    if (random_problem) {
      overlays.set(mesh, imc_state.get_step(), rank, mpi_info.get_n_rank());
    }

    // update time for next step
    imc_state.next_time_step();
  }
}

#endif // replicated_driver_h_

//---------------------------------------------------------------------------//
// end of replicated_driver.h
//---------------------------------------------------------------------------//
