//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   main.cc
 * \author Alex Long
 * \date   July 24 2014
 * \brief  Reads input file, sets up mesh and runs transport
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <mpi.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#include "constants.h"
#include "imc_parameters.h"
#include "imc_state.h"
#include "info.h"
#include "input.h"
#include "mesh.h"
#include "mpi_types.h"
#include "replicated_driver.h"
#include "timer.h"
#include "random_overlay.h"
#include "time_overlay.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // check to see if number of arguments is correct
  if (argc != 2) {
    cout << "Usage: BRANSON <path_to_input_file>" << endl;
    exit(EXIT_FAILURE);
  }

  // wrap main loop scope so objects are destroyed before mpi_finalize is called
  {
    // get MPI parmeters and set them in mpi_info
    const Info mpi_info;
    if (mpi_info.get_rank() == 0) {
      cout << "----- Branson LITE, a massively parallel proxy app for Implicit "
              "Monte Carlo ----"
           << endl;
      cout << "-------- Author: Alex Long (along@lanl.gov) "
              "------------------------------------"
           << endl;
      cout << "-------- Version: 0.81"
              "----------------------------------------------------------"
           << endl
           << endl;
      cout << " Branson compiled on: " << mpi_info.get_machine_name() << endl;
    }

    // make MPI types object
    MPI_Types mpi_types;

    // get input object from filename
    std::string filename(argv[1]);
    Input input(filename, mpi_types);
    int num_realizations, realization_print; // used for random problem
    bool random_problem = input.get_random_problem();
    bool write_output;
    if (random_problem) {
      cout << "DETECTED RANDOM PROBLEM" << endl << endl;
      num_realizations = input.get_num_realizations();
      realization_print = input.get_realization_print();
      write_output = false;
    } else {
      num_realizations = 1;
      write_output = true;
    }
    if (mpi_info.get_rank() == 0)
      input.print_problem_info();

    // timing
    Timer timers;

    // Overlay structure for random problems
    Time_Overlay time_overlay(input);

    for (int i = 0; i < num_realizations; i++) {
      if (random_problem) {
        // re-generate geometry
        input.generate_geometry();
      }

      // IMC paramters setup
      IMC_Parameters imc_p(input);

      // IMC state setup
      IMC_State imc_state(input, mpi_info.get_rank());

      if (!random_problem) {
        // make mesh from input object
        timers.start_timer("Total setup");
      }

      Mesh mesh(input, mpi_types, mpi_info, imc_p);
      mesh.initialize_physical_properties(input);

      if (!random_problem) {
        timers.stop_timer("Total setup");
      }

      MPI_Barrier(MPI_COMM_WORLD);
      // print_MPI_out(mesh, rank, n_rank);

      //--------------------------------------------------------------------------//
      // TRT PHYSICS CALCULATION
      //--------------------------------------------------------------------------//

      if (!random_problem)
        timers.start_timer("Total transport");

      imc_replicated_driver(mesh, imc_state, imc_p, mpi_types, mpi_info, write_output, random_problem, time_overlay);

      if (random_problem) {
        MPI_Barrier(MPI_COMM_WORLD);
        if ((i + 1) % realization_print == 0) {
          cout << "Realization number " << (i + 1) << endl;
        }
      }

      if (!random_problem)
        timers.stop_timer("Total transport");
    }

    if (!random_problem && mpi_info.get_rank() == 0) {
      cout << "****************************************";
      cout << "****************************************" << endl;
      timers.print_timers();
    }

    if (random_problem) {
      time_overlay.print_all(mpi_info.get_rank());
    }

  } // end main loop scope, objects destroyed here

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
}
//---------------------------------------------------------------------------//
// end of main.cc
//---------------------------------------------------------------------------//
