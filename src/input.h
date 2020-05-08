//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   input.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Reads data from XML input file and makes mesh skeleton
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef input_h_
#define input_h_

#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <pugixml.hpp>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#include "config.h"
#include "constants.h"
#include "mpi.h"
#include "mpi_types.h"
#include "region.h"

#include "RNG.h"
#include "geometry_chord_models.h"

//==============================================================================
/*!
 * \class Input
 * \brief Reads input data from an XML file and stores that data
 *
 * Pugi's XML parser is used to read an XML input file. This class stores that
 * information and provides functions to access it. This class also prints the
 * problem information.
 */
//==============================================================================
class Input {
public:
  //! Constructor
  Input(std::string fileName, const MPI_Types &mpi_types) {
    using Constants::ELEMENT;
    using Constants::REFLECT;
    using Constants::VACUUM;
    using Constants::X_NEG;
    using Constants::X_POS;
    using Constants::Y_NEG;
    using Constants::Y_POS;
    using Constants::Z_NEG;
    using Constants::Z_POS;
    using std::cout;
    using std::endl;

    // root rank reads file, prints warnings, and broadcasts to others
    MPI_Region = mpi_types.get_region_type();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      uint32_t x_key, y_key, z_key, key;
      // initialize nunmber of divisions in each dimension to zero
      uint32_t nx_divisions = 0;
      uint32_t ny_divisions = 0;
      uint32_t nz_divisions = 0;
      uint32_t region_ID;

      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> z;

      pugi::xml_document doc;
      pugi::xml_parse_result load_result = doc.load_file(fileName.c_str());

      // error checking
      if (!load_result) {
        cout << load_result.description() << endl;
        cout << "Improperly formatted xml file" << endl;
        pugi::xml_node proto_node = doc.child("prototype");
        cout << "parsed regions: " << endl;
        for (pugi::xml_node_iterator it = proto_node.begin();
             it != proto_node.end(); ++it)
          cout << it->name() << endl;
        exit(EXIT_FAILURE);
      }

      // check each necessary branch exists and create nodes for each
      pugi::xml_node settings_node = doc.child("prototype").child("common");
      pugi::xml_node debug_node = doc.child("prototype").child("debug_options");
      // One of these spatial input types should be specified, but not both!
      pugi::xml_node spatial_node = doc.child("prototype").child("spatial");
      pugi::xml_node simple_spatial_node =
          doc.child("prototype").child("simple_spatial");
      pugi::xml_node random_spatial_node =
          doc.child("prototype").child("random_spatial");
      pugi::xml_node bc_node = doc.child("prototype").child("boundary");
      pugi::xml_node region_node = doc.child("prototype").child("regions");

      if (!settings_node) {
        std::cout << "'common' section not found!" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!spatial_node && !simple_spatial_node && !random_spatial_node) {
        std::cout << "'spatial' or 'simple_spatial' or 'random_spatial' section not found!"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      if ((spatial_node && simple_spatial_node) || (spatial_node && random_spatial_node) && (simple_spatial_node && random_spatial_node)) {
        std::cout
            << "Cannot specify more than one spatial section!"
            << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!bc_node && !random_spatial_node) {
        std::cout << "'boundary' section not found!" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!region_node && !random_spatial_node) {
        std::cout << "'regions' section not found!" << std::endl;
        exit(EXIT_FAILURE);
      }

      tFinish = settings_node.child("t_stop").text().as_double();
      dt = settings_node.child("dt_start").text().as_double();
      tStart = settings_node.child("t_start").text().as_double();
      tMult = settings_node.child("t_mult").text().as_double();
      dtMax = settings_node.child("dt_max").text().as_double();
      unsigned long long n_photons_long =
          settings_node.child("photons").text().as_llong();
      if (n_photons_long > UINT64_MAX) {
        cout << "ERROR: Can't convert " << n_photons_long;
        cout << " to uint64, too large" << endl;
        exit(EXIT_FAILURE);
      }
      n_photons = static_cast<uint64_t>(n_photons_long);
      seed = settings_node.child("seed").text().as_int();
      output_freq = settings_node.child("output_frequency").text().as_int();

      // use particle combing population control
      std::string tempString = settings_node.child_value("use_combing");
      if (tempString == "FALSE")
        use_comb = 0;
      else if (tempString == "TRUE")
        use_comb = 1;
      else {
        cout << "\"use_combing\" not found or recognized, defaulting to TRUE"
             << endl;
        use_comb = 1;
      }

      // write silo flag
      write_silo = false;
      tempString = settings_node.child_value("write_silo");
      if (tempString == "TRUE")
        write_silo = true;

      // use tilting (not currently used)
      use_tilt = false;
      tempString = settings_node.child_value("tilt");
      if (tempString == "TRUE")
        use_tilt = true;

      // debug options
      print_verbose = false;
      print_mesh_info = false;
      tempString = debug_node.child_value("print_verbose");
      if (tempString == "TRUE")
        print_verbose = true;
      tempString = debug_node.child_value("print_mesh_info");
      if (tempString == "TRUE")
        print_mesh_info = true;

      // spatial inputs
      for (pugi::xml_node_iterator it = spatial_node.begin();
           it != spatial_node.end(); ++it) {
        std::string name_string = it->name();
        double d_x_start, d_x_end, d_y_start, d_y_end, d_z_start, d_z_end;
        uint32_t d_x_cells, d_y_cells, d_z_cells;
        if (name_string == "x_division") {
          // x information for this region
          d_x_start = it->child("x_start").text().as_double();
          d_x_end = it->child("x_end").text().as_double();
          d_x_cells = it->child("n_x_cells").text().as_int();
          x_start.push_back(d_x_start);
          x_end.push_back(d_x_end);
          n_x_cells.push_back(d_x_cells);
          nx_divisions++;
          // push back the master x points for silo
          for (uint32_t i = 0; i < d_x_cells; ++i)
            x.push_back(d_x_start + i * (d_x_end - d_x_start) / d_x_cells);
        }

        if (name_string == "y_division") {
          // y information for this region
          d_y_start = it->child("y_start").text().as_double();
          d_y_end = it->child("y_end").text().as_double();
          d_y_cells = it->child("n_y_cells").text().as_int();
          y_start.push_back(d_y_start);
          y_end.push_back(d_y_end);
          n_y_cells.push_back(d_y_cells);
          ny_divisions++;
          // push back the master y points for silo
          for (uint32_t i = 0; i < d_y_cells; ++i)
            y.push_back(d_y_start + i * (d_y_end - d_y_start) / d_y_cells);
        }

        if (name_string == "z_division") {
          // z information for this region
          d_z_start = it->child("z_start").text().as_double();
          d_z_end = it->child("z_end").text().as_double();
          d_z_cells = it->child("n_z_cells").text().as_int();
          z_start.push_back(d_z_start);
          z_end.push_back(d_z_end);
          n_z_cells.push_back(d_z_cells);
          nz_divisions++;
          // push back the master z points for silo
          for (uint32_t i = 0; i < d_z_cells; ++i)
            z.push_back(d_z_start + i * (d_z_end - d_z_start) / d_z_cells);
        }

        if (name_string == "region_map") {
          x_key = it->child("x_div_ID").text().as_int();
          y_key = it->child("y_div_ID").text().as_int();
          z_key = it->child("z_div_ID").text().as_int();
          region_ID = it->child("region_ID").text().as_int();
          // make a unique key using the division ID of x,y and z
          // this mapping allows for 1000 unique divisions in
          // each dimension (way too many)
          key = z_key * 1000000 + y_key * 1000 + x_key;
          region_map[key] = region_ID;
        }
      }

      // spatial inputs
      if (simple_spatial_node) {
        double d_x_start, d_x_end, d_y_start, d_y_end, d_z_start, d_z_end;
        uint32_t d_x_cells, d_y_cells, d_z_cells;

        nx_divisions = 1;
        ny_divisions = 1;
        nz_divisions = 1;

        d_x_start = simple_spatial_node.child("x_start").text().as_double();
        d_x_end = simple_spatial_node.child("x_end").text().as_double();
        d_x_cells = simple_spatial_node.child("n_x_cells").text().as_int();

        // push back the master x points for silo
        for (uint32_t i = 0; i < d_x_cells; ++i)
          x.push_back(d_x_start + i * (d_x_end - d_x_start) / d_x_cells);

        d_y_start = simple_spatial_node.child("y_start").text().as_double();
        d_y_end = simple_spatial_node.child("y_end").text().as_double();
        d_y_cells = simple_spatial_node.child("n_y_cells").text().as_int();

        // push back the master y points for silo
        for (uint32_t i = 0; i < d_y_cells; ++i)
          y.push_back(d_y_start + i * (d_y_end - d_y_start) / d_y_cells);

        d_z_start = simple_spatial_node.child("z_start").text().as_double();
        d_z_end = simple_spatial_node.child("z_end").text().as_double();
        d_z_cells = simple_spatial_node.child("n_z_cells").text().as_int();

        // push back the master z points for silo
        for (uint32_t i = 0; i < d_z_cells; ++i)
          z.push_back(d_z_start + i * (d_z_end - d_z_start) / d_z_cells);

        region_ID = simple_spatial_node.child("region_ID").text().as_int();

        x_start.push_back(d_x_start);
        x_end.push_back(d_x_end);
        n_x_cells.push_back(d_x_cells);
        y_start.push_back(d_y_start);
        y_end.push_back(d_y_end);
        n_y_cells.push_back(d_y_cells);
        z_start.push_back(d_z_start);
        z_end.push_back(d_z_end);
        n_z_cells.push_back(d_z_cells);

        // single region -- maps to 0:
        region_map[0] = region_ID;
      }

      // spatial inputs
      if (random_spatial_node) {
        int geometry_seed;
        double d_x_end;
        int id;
        double chord_start, chord_end, quad_mult, opacA, opacB, opacC, opacS, temp_e, temp_r, dens, spec_heat;
        std::string quad_minmax;
        random_problem = true;

        ny_divisions = 1;
        nz_divisions = 1;

        for (pugi::xml_node_iterator it = random_spatial_node.begin(); it != random_spatial_node.end(); it++) {
          std::string name_string = it->name();
          if (name_string == "problem") {
            num_materials = it->child("materials").text().as_int();
            problem_dist = it->child("distance").text().as_double();
            column_size = it->child("column_size").text().as_double();
            chord_model = it->child("chord_model").text().as_string();
            geometry_seed = it->child("geometry_seed").text().as_int();
            sublayer_cells = it->child("sublayer_cells").text().as_int();
            structured_cells = it->child("structured_cells").text().as_int();
            piecewise_segments = it->child("piecewise_segments").text().as_int();
            num_realizations = it->child("realizations").text().as_int();
            realization_print = it->child("realization_print").text().as_int();
          }

          if (name_string == "material") {
            id = it->child("ID").text().as_int();
            chord_start = it->child("chord_start").text().as_double();
            chord_end = it->child("chord_end").text().as_double();
            quad_minmax = it->child("quad").text().as_string();
            quad_mult = it->child("quad_mult").text().as_double();
            opacA = it->child("opacA").text().as_double();
            opacB = it->child("opacB").text().as_double();
            opacC = it->child("opacC").text().as_double();
            opacS = it->child("opacS").text().as_double();
            temp_e = it->child("initial_T_e").text().as_double();
            temp_r = it->child("initial_T_r").text().as_double();
            dens = it->child("density").text().as_double();
            spec_heat = it->child("CV").text().as_double();
            mat_id.push_back(id);
            mat_chord_start.push_back(chord_start);
            mat_chord_end.push_back(chord_end);
            mat_quad_minmax.push_back(quad_minmax);
            mat_opacA.push_back(opacA);
            mat_opacB.push_back(opacB);
            mat_opacC.push_back(opacC);
            mat_opacS.push_back(opacS);
            mat_T_e.push_back(temp_e);
            mat_T_r.push_back(temp_r);
            mat_dens.push_back(dens);
            mat_CV.push_back(spec_heat);
          }
        }

        if ((mat_id.size() != 2) || (mat_id.size() != num_materials)) {
          cout << "Unsupported/inconsistent number of materials; please use a binary system" << endl;
          exit(EXIT_FAILURE);
        }

        // Establish regions from material data
        for (int i = 0; i < num_materials; i++) {
          Region random_region;
          random_region.set_ID(mat_id.at(i));
          random_region.set_cV(mat_CV.at(i));
          random_region.set_rho(mat_dens.at(i));
          random_region.set_opac_A(mat_opacA.at(i));
          random_region.set_opac_B(mat_opacB.at(i));
          random_region.set_opac_C(mat_opacC.at(i));
          random_region.set_opac_S(mat_opacS.at(i));
          random_region.set_T_e(mat_T_e.at(i));
          random_region.set_T_r(mat_T_r.at(i));
          // map user defined ID to index in region vector
          region_ID_to_index[random_region.get_ID()] = regions.size();
          // add to list of regions
          regions.push_back(random_region);
        }

        // random geometry seed
        seed_g_rng(geometry_seed);

        // placeholder initial geometry - program will re-generate at start of every realization in a random problem
        generate_geometry();
      } else {
        random_problem = false;
        num_realizations = 0;
        realization_print = 0;
        num_materials = 0;
        column_size = 0.0;
        problem_dist = 0.0;
        sublayer_cells = 0;
        structured_cells = 0;
        piecewise_segments = 0;
      }

      if (!random_problem) {
        // read in boundary conditions
        bool b_error = false;
        tempString = bc_node.child_value("bc_right");
        if (tempString == "REFLECT")
          bc[X_POS] = REFLECT;
        else if (tempString == "VACUUM")
          bc[X_POS] = VACUUM;
        else
          b_error = true;

        tempString = bc_node.child_value("bc_left");
        if (tempString == "REFLECT")
          bc[X_NEG] = REFLECT;
        else if (tempString == "VACUUM")
          bc[X_NEG] = VACUUM;
        else
          b_error = true;

        tempString = bc_node.child_value("bc_up");
        if (tempString == "REFLECT")
          bc[Y_POS] = REFLECT;
        else if (tempString == "VACUUM")
          bc[Y_POS] = VACUUM;
        else
          b_error = true;

        tempString = bc_node.child_value("bc_down");
        if (tempString == "REFLECT")
          bc[Y_NEG] = REFLECT;
        else if (tempString == "VACUUM")
          bc[Y_NEG] = VACUUM;
        else
          b_error = true;

        tempString = bc_node.child_value("bc_top");
        if (tempString == "REFLECT")
          bc[Z_POS] = REFLECT;
        else if (tempString == "VACUUM")
          bc[Z_POS] = VACUUM;
        else
          b_error = true;

        tempString = bc_node.child_value("bc_bottom");
        if (tempString == "REFLECT")
          bc[Z_NEG] = REFLECT;
        else if (tempString == "VACUUM")
          bc[Z_NEG] = VACUUM;
        else
          b_error = true;

        if (b_error) {
          cout << "ERROR: Boundary type not recognized. Exiting..." << endl;
          exit(EXIT_FAILURE);
        }
      } else {
        bc[X_POS] = VACUUM;
        bc[X_NEG] = VACUUM;
        bc[Y_POS] = REFLECT;
        bc[Y_NEG] = REFLECT;
        bc[Z_POS] = REFLECT;
        bc[Z_NEG] = REFLECT;
      } // end boundary node

      // read in region data
      if (!random_problem) {
        for (pugi::xml_node_iterator it = region_node.begin();
            it != region_node.end(); ++it) {
          std::string name_string = it->name();
          if (name_string == "region") {
            Region temp_region;
            temp_region.set_ID(it->child("ID").text().as_int());
            temp_region.set_cV(it->child("CV").text().as_double());
            temp_region.set_rho(it->child("density").text().as_double());
            temp_region.set_opac_A(it->child("opacA").text().as_double());
            temp_region.set_opac_B(it->child("opacB").text().as_double());
            temp_region.set_opac_C(it->child("opacC").text().as_double());
            temp_region.set_opac_S(it->child("opacS").text().as_double());
            temp_region.set_T_e(it->child("initial_T_e").text().as_double());
            // default T_r to T_e if not specified
            temp_region.set_T_r(it->child("initial_T_r").text().as_double());
            // map user defined ID to index in region vector
            region_ID_to_index[temp_region.get_ID()] = regions.size();
            // add to list of regions
            regions.push_back(temp_region);
          }
        }
      }

      if (!random_problem) {
        division_set(nx_divisions, ny_divisions, nz_divisions);

        // append the last point values and allocate SILO array
        x.push_back(x_end.back());
        y.push_back(y_end.back());
        z.push_back(z_end.back());
        silo_x = std::vector<float>(x.size());
        silo_y = std::vector<float>(y.size());
        silo_z = std::vector<float>(z.size());
        for (uint32_t i = 0; i < x.size(); ++i)
          silo_x[i] = x[i];
        for (uint32_t j = 0; j < y.size(); ++j)
          silo_y[j] = y[j];
        for (uint32_t k = 0; k < z.size(); ++k)
          silo_z[k] = z[k];
      }
    } // end xml parse on root rank

    broadcast();
  }

  //! Destructor
  ~Input(){ delete g_rng; }

  void broadcast() {
    using std::vector;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n_bools = 7;
    const int n_uint = 13;
    const int n_uint64 = 3;
    const int n_doubles = 8;

    // root rank broadcasts read values
    if (rank == 0) {
      // some helper values
      uint32_t n_regions = regions.size();
      uint32_t n_x_div = n_x_cells.size();
      uint32_t n_y_div = n_y_cells.size();
      uint32_t n_z_div = n_z_cells.size();

      // bools
      vector<int> all_bools = {write_silo, use_tilt,      use_comb,
                               use_strat,  print_verbose, print_mesh_info, random_problem};
      MPI_Bcast(&all_bools[0], n_bools, MPI_INT, 0, MPI_COMM_WORLD);

      // bcs
      vector<int> bcast_bcs = {bc[0], bc[1], bc[2], bc[3], bc[4], bc[5]};
      MPI_Bcast(&bcast_bcs[0], 6, MPI_INT, 0, MPI_COMM_WORLD);

      // uint32
      vector<uint32_t> all_uint = {seed,
                                   output_freq,
                                   n_divisions,
                                   n_global_x_cells,
                                   n_global_y_cells,
                                   n_global_z_cells,
                                   n_regions,
                                   n_x_div,
                                   n_y_div,
                                   n_z_div,
                                   num_materials,
                                   sublayer_cells,
                                   structured_cells};
      MPI_Bcast(&all_uint[0], n_uint, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

      // uint64
      vector<uint64_t> all_uint64 = {n_photons,
                                     num_realizations,
                                     realization_print};
      MPI_Bcast(&all_uint64[0], n_uint64, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

      // double
      vector<double> all_doubles = {tStart, dt, tFinish,
                                    tMult, dtMax, T_source, column_size, problem_dist};
      MPI_Bcast(&all_doubles[0], n_doubles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // region processing
      MPI_Bcast(&regions[0], n_regions, MPI_Region, 0, MPI_COMM_WORLD);
      vector<uint32_t> division_key;
      vector<uint32_t> region_at_division;
      for (auto rmap : region_map) {
        division_key.push_back(rmap.first);
        region_at_division.push_back(rmap.second);
      }
      if (division_key.size() != n_divisions ||
          region_at_division.size() != n_divisions)
        std::cout << "something went wrong in division key communication"
                  << std::endl;

      MPI_Bcast(&division_key[0], n_divisions, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&region_at_division[0], n_divisions, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);

      // mesh spacing and coordinate processing
      MPI_Bcast(&x_start[0], n_x_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&x_end[0], n_x_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_x_cells[0], n_x_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y_start[0], n_y_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y_end[0], n_y_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_y_cells[0], n_y_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&z_start[0], n_z_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&z_end[0], n_z_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_z_cells[0], n_z_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    } else {
      // set bools
      vector<int> all_bools(7);

      MPI_Bcast(&all_bools[0], n_bools, MPI_INT, 0, MPI_COMM_WORLD);
      write_silo = all_bools[0];
      use_tilt = all_bools[1];
      use_comb = all_bools[2];
      use_strat = all_bools[3];
      print_verbose = all_bools[4];
      print_mesh_info = all_bools[5];
      random_problem = all_bools[6];

      // set bcs
      vector<int> bcast_bcs(6);
      MPI_Bcast(&bcast_bcs[0], 6, MPI_INT, 0, MPI_COMM_WORLD);
      for (int i = 0; i < 6; ++i)
        bc[i] = Constants::bc_type(bcast_bcs[i]);

      // set uints
      vector<uint32_t> all_uint(n_uint);
      MPI_Bcast(&all_uint[0], n_uint, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      seed = all_uint[0];
      output_freq = all_uint[1];
      n_divisions = all_uint[2];
      n_global_x_cells = all_uint[3];
      n_global_y_cells = all_uint[4];
      n_global_z_cells = all_uint[5];
      uint32_t n_regions = all_uint[6];
      uint32_t n_x_div = all_uint[7];
      uint32_t n_y_div = all_uint[8];
      uint32_t n_z_div = all_uint[9];
      num_materials = all_uint[10];
      sublayer_cells = all_uint[11];
      structured_cells = all_uint[12];

      // uint64
      vector<uint64_t> all_uint64(n_uint64);
      MPI_Bcast(&all_uint64[0], n_uint64, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      n_photons = all_uint64[0];
      num_realizations = all_uint64[1];
      realization_print = all_uint64[2];

      vector<double> all_doubles(n_doubles);
      MPI_Bcast(&all_doubles[0], n_doubles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      tStart = all_doubles[0];
      dt = all_doubles[1];
      tFinish = all_doubles[2];
      tMult = all_doubles[3];
      dtMax = all_doubles[4];
      T_source = all_doubles[5];
      column_size = all_doubles[6];
      problem_dist = all_doubles[7];

      // region processing (broadcast directly into member variable)
      regions.resize(n_regions);
      MPI_Bcast(&regions[0], n_regions, MPI_Region, 0, MPI_COMM_WORLD);

      n_divisions = all_uint[2];
      vector<uint32_t> division_key(n_divisions);
      vector<uint32_t> region_at_division(n_divisions);
      MPI_Bcast(&division_key[0], n_divisions, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&region_at_division[0], n_divisions, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);
      for (uint32_t i = 0; i < n_divisions; ++i) {
        region_map[division_key[i]] = region_at_division[i];
      }
      for (uint32_t i = 0; i < n_regions; ++i)
        region_ID_to_index[regions[i].get_ID()] = i;

      // mesh spacing and coordinate processing
      x_start.resize(n_x_div);
      x_end.resize(n_x_div);
      n_x_cells.resize(n_x_div);

      y_start.resize(n_y_div);
      y_end.resize(n_y_div);
      n_y_cells.resize(n_y_div);

      z_start.resize(n_z_div);
      z_end.resize(n_z_div);
      n_z_cells.resize(n_z_div);

      MPI_Bcast(&x_start[0], n_x_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&x_end[0], n_x_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_x_cells[0], n_x_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y_start[0], n_y_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y_end[0], n_y_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_y_cells[0], n_y_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(&z_start[0], n_z_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&z_end[0], n_z_div, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_z_cells[0], n_z_div, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

      // other ranks don't write silo so they don't need the x,y,z coordinates
      silo_x = std::vector<float>(n_global_x_cells, 0.0);
      silo_y = std::vector<float>(n_global_y_cells, 0.0);
      silo_z = std::vector<float>(n_global_z_cells, 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  //! Print the information read from the input file
  void print_problem_info(void) const {
    using Constants::a;
    using Constants::c;
    using std::cout;
    using std::endl;

    cout << "Problem Specifications:";
    cout << "Constants -- c: " << c << " (cm/sh) , a: " << a << endl;
    cout << "Run Parameters-- Photons: " << n_photons
         << ", time finish: " << tFinish;
    cout << " (sh), time step: " << dt << " (sh) ," << endl;

    cout << " time multiplier: " << tMult << " , max dt:" << dtMax;
    cout << " (sh), Random number seed: " << seed;
    cout << " , output frequency: " << output_freq << endl;

    // cout<<"Sampling -- Emission Position: ";
    // if (use_tilt) cout<<"source tilting (x only), ";
    // else cout<<"uniform (default), ";
    // cout<<" Angle: ";
    cout << "Stratified sampling in angle: ";
    if (use_strat)
      cout << "TRUE" << endl;
    else
      cout << "FALSE" << endl;

    if (use_comb)
      cout << "Combing census enabled (default)" << endl;
    else
      cout << "No combing" << endl;

    if (print_verbose)
      cout << "Verbose printing mode enabled" << endl;
    else
      cout << "Terse printing mode (default)" << endl;

#ifdef VIZ_LIBRARIES_FOUND
    if (write_silo)
      cout << "SILO output enabled" << endl;
    else
      cout << "SILO output disabled (default)" << endl;
#else
    if (write_silo)
      cout << "NOTE: SILO libraries not linked... will attempt HDF5" << endl;
    else
      cout << "No visualization set" << endl;
#endif
    cout << "Spatial Information -- cells x,y,z: " << n_global_x_cells << " ";
    cout << n_global_y_cells << " " << n_global_z_cells << endl;

    cout << "--Material Information--" << endl;
    for (uint32_t r = 0; r < regions.size(); ++r) {
      cout << " heat capacity: " << regions[r].get_cV();
      cout << " opacity constants: " << regions[r].get_opac_A() << " + "
           << regions[r].get_opac_B() << "^" << regions[r].get_opac_C();
      cout << ", scattering opacity: " << regions[r].get_scattering_opacity()
           << endl;
    }

    cout << "--Parallel Information--" << endl;
    cout << "REPLICATED" << endl;
    cout << "No parameters are needed in replicated mode";
    cout << endl;

    cout << endl;
  }

  void generate_geometry(void) {
    using std::cout;
    using std::endl;

    int rank;
    bool exit_call = false;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      if (chord_model == "constant")
        generate_geometry_constant();
      else if (chord_model == "linear")
        generate_geometry_linear();
      else if (chord_model == "linear_direct")
        generate_geometry_linear_direct();
      else if (chord_model == "linear_piecewise")
        generate_geometry_linear_piecewise();
      else if (chord_model == "linear_piecewise_direct")
        generate_geometry_linear_piecewise_direct();
      else if (chord_model == "quadratic")
        generate_geometry_quadratic();
      else if (chord_model == "quadratic_piecewise")
        generate_geometry_quadratic_piecewise();
      else if (chord_model == "quadratic_piecewise_direct")
        generate_geometry_quadratic_piecewise_direct();
      else {
        cout << "ERROR: Material chord model not recognized." << endl;
        cout << "Options are:" << endl;
        cout << "linear" << endl;
        cout << "linear_direct" << endl;
        cout << "linear_piecewise" << endl;
        cout << "linear_piecewise_direct" << endl;
        cout << "quadratic" << endl;
        cout << "quadratic_piecewise" << endl;
        cout << "quadratic_piecewise_direct" << endl;
        cout << "\nExiting..." << endl;
        exit_call = true;
      }
    }
    MPI_Bcast(&exit_call, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_call)
      exit(EXIT_FAILURE);

    broadcast();
  }

  void generate_geometry_constant(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord;
    int material_num;

    material_num = (rand_num < mat_chord_start.at(0) / (mat_chord_start.at(0) + mat_chord_start.at(1))) ? 0 : 1;

    y.push_back(0.0);
    z.push_back(0.0);

    while (cons_dist < problem_dist) {
      x_start.push_back(cons_dist);
      // Generate a random number
      rand_num = g_rng->generate_random_number();
      chord = mat_chord_start.at(material_num);
      dist -= log(rand_num) * chord;
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);

      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_linear_direct(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // Problem parameters
    std::vector<float> chord_slope;
    for (int m = 0; m < num_materials; m++) {
      chord_slope.push_back((mat_chord_end.at(m) - mat_chord_start.at(m)) / problem_dist);
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord_start, chord_end;
    int material_num;

    // starting material volume fraction is set by initial chord lengths
    material_num = (rand_num < mat_chord_start.at(0) / (mat_chord_start.at(0) + mat_chord_start.at(1))) ? 0 : 1;

    y.push_back(0.0);
    z.push_back(0.0);

    while (cons_dist < problem_dist) {
      x_start.push_back(cons_dist);
      // Generate a random number
      rand_num = g_rng->generate_random_number();
      chord_start = mat_chord_start.at(material_num);
      chord_end = mat_chord_end.at(material_num);

      // Calculate and append the material length
      dist = (pow(rand_num, -chord_slope.at(material_num)) * (chord_start + chord_slope.at(material_num) * cons_dist) - chord_start - chord_slope.at(material_num) * cons_dist) / chord_slope.at(material_num);

      cons_dist += dist;

      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);

      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_linear(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // Problem parameters
    std::vector<double> chord_slope, limiting_value_chord;
    for (int m = 0; m < num_materials; m++) {
      chord_slope.push_back((mat_chord_end.at(m) - mat_chord_start.at(m)) / problem_dist);
      if (chord_slope.at(m) > 0.0)
        limiting_value_chord.push_back(mat_chord_start.at(m));
      else
        limiting_value_chord.push_back(mat_chord_end.at(m));
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord_start, chord_end;
    int material_num;
    double prob_accept, buffer_chord;

    // starting material volume fraction is set by initial chord lengths
    material_num = (rand_num < mat_chord_start.at(0) / (mat_chord_start.at(0) + mat_chord_start.at(1))) ? 0 : 1;

    y.push_back(0.0);
    z.push_back(0.0);

    while (cons_dist < problem_dist) {
      dist = 0.0;
      x_start.push_back(cons_dist);
      chord_start = mat_chord_start.at(material_num);
      chord_end = mat_chord_end.at(material_num);

      // Loop for rejection sampling
      bool accepted = false;
      while (!accepted) {
        // Generate a random number
        rand_num = g_rng->generate_random_number();

        // Sample from a homogeneous distribution of intensity equal to maximum chord
        dist -= limiting_value_chord.at(material_num) * log(rand_num);
        if (dist > problem_dist) break;

        // Maximum value achieved with minimum length chord
        // Or, conversely, the inverse of the chord (therefore inverse division)
        buffer_chord = linear_chord(chord_start, chord_slope.at(material_num), cons_dist + dist);
        prob_accept = limiting_value_chord.at(material_num) / buffer_chord;

        rand_num = g_rng->generate_random_number();
        if (rand_num < prob_accept)
          accepted = true;
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);
      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_linear_piecewise_direct(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord_start, chord_end;
    int material_num;

    // Iteration values
    bool accepted;
    int segment, current_segment;
    double first_portion, middle_portion;

    // Determine first material to use
    // For piecewise constant model, initial distance is at 0.0 and so the probability is equivalent to the term ratio of the averages of the first segments
    std::vector<double> first_value;
    for (int m = 0; m < num_materials; m++) {
      first_value.push_back(piecewise_constant_linear_chord(mat_chord_start.at(m), mat_chord_end.at(m), piecewise_segments, problem_dist, 0.0));
    }
    material_num = (rand_num < first_value.at(0) / (first_value.at(0) + first_value.at(1))) ? 0 : 1;

    // Delta distance in piecewise function (required to obtain chords at each segment)
    double delta_dist = problem_dist / static_cast<double>(piecewise_segments);

    while (cons_dist < problem_dist) {
      // Generate a random number
      g_rng->generate_random_number();

      // Assign a chord length based on material number
      chord_start = mat_chord_start.at(material_num);
      chord_end = mat_chord_end.at(material_num);

      // Loop for sampling distance (required to test each segment of piecewise function)
      // Note that segments are zero-indexed
      accepted = false;
      // Start at current segment
      current_segment = static_cast<int>(cons_dist / delta_dist);
      segment = current_segment;
      while (!accepted) {
        if (segment > current_segment) {
          first_portion = ((current_segment + 1) * delta_dist - cons_dist) / piecewise_constant_linear_chord(chord_start, chord_end, piecewise_segments, problem_dist, cons_dist);
          middle_portion = 0.0;
          for (int k = current_segment + 1; k < segment; k++) {
            middle_portion += delta_dist / piecewise_constant_linear_chord(chord_start, chord_end, piecewise_segments, problem_dist, (static_cast<double>(k) + 0.5) * delta_dist);
          }
          dist = segment * delta_dist - piecewise_constant_linear_chord(chord_start, chord_end, piecewise_segments, problem_dist, (static_cast<double>(segment) + 0.5) * delta_dist) * (log(rand_num) + middle_portion + first_portion) - cons_dist;
        } else {
          dist = -log(rand_num) * piecewise_constant_linear_chord(chord_start, chord_end, piecewise_segments, problem_dist, cons_dist);
        }
        if (dist > (problem_dist - cons_dist))
          dist = problem_dist - cons_dist;
        if ((dist + cons_dist > (segment + 1) * delta_dist) || (dist + cons_dist < segment * delta_dist)) {
          segment++;
        } else {
          accepted = true;
          // Test for negative
          if (dist < 0.0) {
            accepted = false;
            dist = 0.0;
            rand_num = g_rng->generate_random_number();
          }
        }
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);
      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_linear_piecewise(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // The value that the chord possesses for the maximum value computed (for rejection purposes)
    // As exponential distribution is computed using the inverse of the chrod-length, the minimum value is chosen
    std::vector<double> first_value, end_value, limiting_value_chord;
    for (int m = 0; m < num_materials; m++) {
      first_value.push_back(piecewise_constant_linear_chord(mat_chord_start.at(m), mat_chord_end.at(m), piecewise_segments, problem_dist, 0.0));
      end_value.push_back(piecewise_constant_linear_chord(mat_chord_start.at(m), mat_chord_end.at(m), piecewise_segments, problem_dist, problem_dist));
      limiting_value_chord.push_back(std::min(first_value.at(m), end_value.at(m)));
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord_start, chord_end;
    int material_num;
    double prob_accept, buffer_chord;

    // Determine first material to use
    // For piecewise constant model, initial distance is at 0.0 and so the probability is equivalent to the term ratio of the averages of the first segments
    material_num = (rand_num < first_value.at(0) / (first_value.at(0) + first_value.at(1))) ? 0 : 1;

    while (cons_dist < problem_dist) {
      dist = 0.0;
      // Assign a chord length based on material number
      chord_start = mat_chord_start.at(material_num);
      chord_end = mat_chord_end.at(material_num);

      // Loop for rejection sampling
      bool accepted = false;
      while (!accepted) {
        // Generate a random number
        rand_num = g_rng->generate_random_number();

        // Sample from a homogeneous distribution of intensity equal to maximum chord
        dist -= limiting_value_chord.at(material_num) * log(rand_num);
        if (dist > problem_dist) break;

        // Maximum value achieved with minimum length chord
        // Or, conversely, the inverse of the chord (therefore inverse division)
        buffer_chord = piecewise_constant_linear_chord(chord_start, chord_end, piecewise_segments, problem_dist, cons_dist + dist);
        prob_accept = limiting_value_chord.at(material_num) / buffer_chord;

        rand_num = g_rng->generate_random_number();
        if (rand_num < prob_accept)
          accepted = true;
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);
      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_quadratic(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // Problem parameters
    std::vector<double> mid_value, delta_t, delta_m, param_a, param_b, param_c, limiting_value_chord;
    for (int m = 0; m < num_materials; m++) {
      if (mat_quad_minmax.at(m) == "max") {
        mid_value.push_back(mid_value_max_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else if (mat_quad_minmax.at(m) == "min") {
        mid_value.push_back(mid_value_min_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else {
        std::cout << "Quadratic min/max not recognized in <quad> setting. Please use \"min\" or \"max\". Exiting..." << std::endl;
        exit(EXIT_FAILURE);
      }
      delta_t.push_back(mat_chord_end.at(m) - mat_chord_start.at(m));
      delta_m.push_back(mid_value.at(m) - mat_chord_start.at(m));
      param_c.push_back(calc_c(mat_chord_start.at(m)));
      param_b.push_back(calc_b(delta_m.at(m), delta_t.at(m), problem_dist));
      param_a.push_back(calc_a(delta_t.at(m), param_b.at(m), problem_dist));
      limiting_value_chord.push_back(std::min(mid_value.at(m), std::min(mat_chord_start.at(m), mat_chord_end.at(m))));
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0, chord_start, chord_end;
    int material_num;
    double prob_accept, buffer_chord;

    // Starting material volume fraction is set by initial chord lengths
    material_num = (rand_num < mat_chord_start.at(0) / (mat_chord_start.at(0) + mat_chord_start.at(1))) ? 0 : 1;

    y.push_back(0.0);
    z.push_back(0.0);

    while (cons_dist < problem_dist) {
      dist = 0.0;
      x_start.push_back(cons_dist);
      chord_start = mat_chord_start.at(material_num);
      chord_end = mat_chord_end.at(material_num);

      // Loop for rejection sampling
      bool accepted = false;
      while (!accepted) {
        // Generate a random number
        rand_num = g_rng->generate_random_number();

        // Sample from a homogeneous distribution of intensity equal to maximum chord
        dist -= limiting_value_chord.at(material_num) * log(rand_num);
        if (dist > problem_dist) break;

        // Maximum value achieved with minimum length chord
        // Or, conversely, the inverse of the chord (therefore inverse division)
        buffer_chord = quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), cons_dist + dist);
        prob_accept = limiting_value_chord.at(material_num) / buffer_chord;

        rand_num = g_rng->generate_random_number();
        if (rand_num < prob_accept)
          accepted = true;
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);

      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_quadratic_piecewise_direct(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // Problem parameters
    std::vector<double> mid_value, delta_t, delta_m, param_a, param_b, param_c, first_value;
    for (int m = 0; m < num_materials; m++) {
      if (mat_quad_minmax.at(m) == "max") {
        mid_value.push_back(mid_value_max_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else if (mat_quad_minmax.at(m) == "min") {
        mid_value.push_back(mid_value_min_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else {
        std::cout << "Quadratic min/max not recognized in <quad> setting. Please use \"min\" or \"max\". Exiting..." << std::endl;
        exit(EXIT_FAILURE);
      }
      delta_t.push_back(mat_chord_end.at(m) - mat_chord_start.at(m));
      delta_m.push_back(mid_value.at(m) - mat_chord_start.at(m));
      param_c.push_back(calc_c(mat_chord_start.at(m)));
      param_b.push_back(calc_b(delta_m.at(m), delta_t.at(m), problem_dist));
      param_a.push_back(calc_a(delta_t.at(m), param_b.at(m), problem_dist));
      first_value.push_back(piecewise_constant_quad_chord(param_a.at(m), param_b.at(m), param_c.at(m), piecewise_segments, problem_dist, 0.0));
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0;
    int material_num;

    // Iteration values
    bool accepted;
    int segment, current_segment;
    double first_portion, middle_portion;

    // Determine first material to use
    // For piecewise constant model, initial distance is at 0.0 and so the probability is equivalent to the term ratio of the averages of the first segments
    material_num = (rand_num < first_value.at(0) / (first_value.at(0) + first_value.at(1))) ? 0 : 1;

    // Delta distance in piecewise function (required to obtain chords at each segment)
    double delta_dist = problem_dist / static_cast<double>(piecewise_segments);

    while (cons_dist < problem_dist) {
      // Generate a random number
      g_rng->generate_random_number();

      // Loop for sampling distance (required to test each segment of piecewise function)
      // Note that segments are zero-indexed
      accepted = false;
      // Start at current segment
      current_segment = static_cast<int>(cons_dist / delta_dist);
      segment = current_segment;
      while(!accepted) {
        if (segment > current_segment) {
          first_portion = ((current_segment + 1) * delta_dist - cons_dist) / piecewise_constant_quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), piecewise_segments, problem_dist, cons_dist);
          middle_portion = 0.0;
          for (int k = current_segment + 1; k < segment; k++) {
            middle_portion += delta_dist / piecewise_constant_quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), piecewise_segments, problem_dist, (static_cast<double>(k) + 0.5) * delta_dist);
          }
          dist = segment * delta_dist - piecewise_constant_quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), piecewise_segments, problem_dist, (static_cast<double>(segment) + 0.5) * delta_dist) * (log(rand_num) + middle_portion + first_portion) - cons_dist;
        } else {
          dist = -log(rand_num) * piecewise_constant_quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), piecewise_segments, problem_dist, cons_dist);
        }
        if (dist > (problem_dist - cons_dist))
          dist = problem_dist - cons_dist;
        if ((dist + cons_dist > (segment + 1) * delta_dist) || (dist + cons_dist < segment * delta_dist)) {
          segment++;
        } else {
          accepted = 1;
          // Test for negative
          if (dist < 0.0) {
            accepted = 0;
            dist = 0.0;
            rand_num = g_rng->generate_random_number();
          }
        }
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);
      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void generate_geometry_quadratic_piecewise(void) {
    // clear other/previous geometry data
    x_start.clear();
    x_end.clear();
    y_start.clear();
    y_end.clear();
    z_start.clear();
    z_end.clear();
    n_x_cells.clear();
    n_y_cells.clear();
    n_z_cells.clear();
    silo_x.clear();
    silo_y.clear();
    silo_z.clear();
    region_map.clear();

    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 1;
    uint32_t nz_divisions = 1;

    uint32_t x_key, y_key, z_key, key, region_ID;

    std::vector<float> x, y, z;

    double cur_dist;

    // Problem parameters
    // Limiting value that the chord possesses for the maximum value computed (for rejection purposes)
    // As exponential distribution is computed using the inverse of the chrod-length, the minimum value is chosen
    std::vector<double> mid_value, delta_t, delta_m, param_a, param_b, param_c, first_value, end_value, limiting_value_chord;
    for (int m = 0; m < num_materials; m++) {
      if (mat_quad_minmax.at(m) == "max") {
        mid_value.push_back(mid_value_max_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else if (mat_quad_minmax.at(m) == "min") {
        mid_value.push_back(mid_value_min_func(mat_chord_start.at(m), mat_chord_end.at(m), mat_quad_mult.at(m)));
      } else {
        std::cout << "Quadratic min/max not recognized in <quad> setting. Please use \"min\" or \"max\". Exiting..." << std::endl;
        exit(EXIT_FAILURE);
      }
      delta_t.push_back(mat_chord_end.at(m) - mat_chord_start.at(m));
      delta_m.push_back(mid_value.at(m) - mat_chord_start.at(m));
      param_c.push_back(calc_c(mat_chord_start.at(m)));
      param_b.push_back(calc_b(delta_m.at(m), delta_t.at(m), problem_dist));
      param_a.push_back(calc_a(delta_t.at(m), param_b.at(m), problem_dist));
      first_value.push_back(piecewise_constant_quad_chord(param_a.at(m), param_b.at(m), param_c.at(m), piecewise_segments, problem_dist, 0.0));
      end_value.push_back(piecewise_constant_quad_chord(param_a.at(m), param_b.at(m), param_c.at(m), piecewise_segments, problem_dist, problem_dist));
      limiting_value_chord.push_back(std::min(mid_value.at(m), std::min(first_value.at(m), end_value.at(m))));
    }

    // push back the master x points for silo
    double rand_num = g_rng->generate_random_number();
    double cons_dist = 0.0, dist = 0.0;
    int material_num;
    double prob_accept, buffer_chord;

    // Determine first material to use
    // For piecewise constant model, initial distance is at 0.0 and so the probability is equivalent to the term ratio of the averages of the first segments
    material_num = (rand_num < first_value.at(0) / (first_value.at(0) + first_value.at(1))) ? 0 : 1;

    while (cons_dist < problem_dist) {
      dist = 0.0;

      // Loop for rejection sampling
      bool accepted = false;
      while (!accepted) {
        // Generate a random number
        rand_num = g_rng->generate_random_number();

        // Sample from a homogeneous distribution of intensity equal to maximum chord
        dist -= limiting_value_chord.at(material_num) * log(rand_num);
        if (dist > problem_dist) break;

        // Maximum value achieved with minimum length chord
        // Or, conversely, the inverse of the chord (therefore inverse division)
        buffer_chord = piecewise_constant_quad_chord(param_a.at(material_num), param_b.at(material_num), param_c.at(material_num), piecewise_segments, problem_dist, cons_dist + dist);
        prob_accept = limiting_value_chord.at(material_num) / buffer_chord;

        rand_num = g_rng->generate_random_number();
        if (rand_num < prob_accept)
          accepted = true;
      }
      cons_dist += dist;
      // Check on thickness to not overshoot the boundary
      if (cons_dist > problem_dist) {
        dist += problem_dist - cons_dist;
        cons_dist = problem_dist;
      }
      // Further discretize geometry
      for (uint32_t i = 0; i < sublayer_cells; i++) {
        cur_dist = cons_dist - dist + (dist / static_cast<double>(sublayer_cells) * static_cast<double>(i));
        if (cur_dist != problem_dist)
          x.push_back(cur_dist);
      }
      x_end.push_back(cons_dist);
      n_x_cells.push_back(sublayer_cells);
      x_key = nx_divisions;
      y_key = 0;
      z_key = 0;
      key = z_key * 1000000 + y_key * 1000 + x_key;
      region_map[key] = mat_id.at(material_num);
      nx_divisions++;
      material_num = (material_num == 0) ? 1 : 0;
    }

    x.push_back(problem_dist);
    y_start.push_back(0.0);
    y_end.push_back(column_size);
    n_y_cells.push_back(1);
    z_start.push_back(0.0);
    z_end.push_back(column_size);
    n_z_cells.push_back(1);

    division_set(nx_divisions, ny_divisions, nz_divisions);

    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = std::vector<float>(x.size());
    silo_y = std::vector<float>(y.size());
    silo_z = std::vector<float>(z.size());
    for (uint32_t i = 0; i < x.size(); ++i)
      silo_x[i] = x[i];
    for (uint32_t j = 0; j < y.size(); ++j)
      silo_y[j] = y[j];
    for (uint32_t k = 0; k < z.size(); ++k)
      silo_z[k] = z[k];
  }

  void division_set(uint32_t nx_divisions, uint32_t ny_divisions, uint32_t nz_divisions) {
    n_divisions = nx_divisions * ny_divisions * nz_divisions;

    // get global cell counts
    n_global_x_cells = std::accumulate(n_x_cells.begin(), n_x_cells.end(), 0);
    n_global_y_cells = std::accumulate(n_y_cells.begin(), n_y_cells.end(), 0);
    n_global_z_cells = std::accumulate(n_z_cells.begin(), n_z_cells.end(), 0);

    // make sure at least one region is specified
    if (!regions.size()) {
      std::cout << "ERROR: No regions were specified. Exiting..." << std::endl;
      exit(EXIT_FAILURE);
    }

    // the total number of divisions must equal the number of unique region maps
    if (n_divisions != region_map.size()) {
      std::cout << "ERROR: Number of total divisions must match the number of ";
      std::cout << "unique region maps. Exiting..." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  //! Return the number of global cells in the x direction
  uint32_t get_global_n_x_cells(void) const { return n_global_x_cells; }
  //! Return the number of global cells in the y direction
  uint32_t get_global_n_y_cells(void) const { return n_global_y_cells; }
  //! Return the number of global cells in the z direction
  uint32_t get_global_n_z_cells(void) const { return n_global_z_cells; }

  //! Return the number of x cells in a given x division
  uint32_t get_x_division_cells(const uint32_t &div) const {
    return n_x_cells[div];
  }
  //! Return the number of y cells in a given y division
  uint32_t get_y_division_cells(const uint32_t &div) const {
    return n_y_cells[div];
  }
  //! Return the number of z cells in a given z division
  uint32_t get_z_division_cells(const uint32_t &div) const {
    return n_z_cells[div];
  }

  //! Return a constant reference to the x coordinates of the mesh in SILO format
  const std::vector<float> &get_silo_x(void) const { return silo_x; }
  //! Return a constant reference to the y coordinates of the mesh in SILO format
  const std::vector<float> &get_silo_y(void) const { return silo_y; }
  //! Return a constant reference to the z coordinates of the mesh in SILO format
  const std::vector<float> &get_silo_z(void) const { return silo_z; }

  //! Return the total number of x divisions in the problem
  uint32_t get_n_x_divisions(void) const { return n_x_cells.size(); }
  //! Return the total number of y divisions in the problem
  uint32_t get_n_y_divisions(void) const { return n_y_cells.size(); }
  //! Return the total number of z divisions in the problem
  uint32_t get_n_z_divisions(void) const { return n_z_cells.size(); }

  //! Return the x grid spacing in a given x division
  double get_dx(const uint32_t &div) const {
    return (x_end[div] - x_start[div]) / n_x_cells[div];
  }
  //! Return the y grid spacing in a given y division
  double get_dy(const uint32_t &div) const {
    return (y_end[div] - y_start[div]) / n_y_cells[div];
  }
  //! Return the z grid spacing in a given z division
  double get_dz(const uint32_t &div) const {
    return (z_end[div] - z_start[div]) / n_z_cells[div];
  }

  //! Return the starting x position of a given x division
  double get_x_start(const uint32_t &div) const { return x_start[div]; }
  //! Return the starting y position of a given y division
  double get_y_start(const uint32_t &div) const { return y_start[div]; }
  //! Return the starting z position of a given z division
  double get_z_start(const uint32_t &div) const { return z_start[div]; }

  //! Return the value of the use tilt option
  bool get_tilt_bool(void) const { return use_tilt; }
  //! Return the value of the use comb option
  bool get_comb_bool(void) const { return use_comb; }
  //! Return the value of the stratified option
  bool get_stratified_bool(void) const { return use_strat; }
  //! Return the value of the write SILO option
  bool get_write_silo_bool(void) const { return write_silo; }
  //! Return the value of the verbose printing option
  bool get_verbose_print_bool(void) const { return print_verbose; }
  //! Return the value of the mesh print option
  bool get_print_mesh_info_bool(void) const { return print_mesh_info; }
  //! Return the frequency of timestep summary printing
  uint32_t get_output_freq(void) const { return output_freq; }

  //! Return the timestep size (shakes)
  double get_dt(void) const { return dt; }
  //! Return the starting time (shakes)
  double get_time_start(void) const { return tStart; }
  //! Return the finish time (shakes)
  double get_time_finish(void) const { return tFinish; }
  //! Return the multiplication factor for the timestep
  double get_time_mult(void) const { return tMult; }
  //! Return the maximum timestep size (shakes)
  double get_dt_max(void) const { return dtMax; }
  //! Return the input seed for the RNG
  int get_rng_seed(void) const { return seed; }
  //! Return the number of photons set in the input file to run
  uint64_t get_number_photons(void) const { return n_photons; }

  // source functions
  //! Return the temperature of the face source
  double get_source_T(void) const { return T_source; }

  //! Return the number of material regions
  uint32_t get_n_regions(void) const { return regions.size(); }

  //! Return vector of regions
  const std::vector<Region> &get_regions(void) const { return regions; }

  //! Return region given user set ID
  const Region &get_region(const uint32_t region_ID) const {
    return regions[region_ID_to_index.at(region_ID)];
  }

  //! Return unique index given division indices

  //! Make a unique key using the division ID of x,y and z. This mapping
  //! allows for 1000 unique divisions in each dimension (way too many)
  uint32_t get_region_index(const uint32_t x_div, const uint32_t y_div,
                            const uint32_t z_div) const {
    const uint32_t key = z_div * 1000000 + y_div * 1000 + x_div;
    return region_ID_to_index.at(region_map.at(key));
  }

  //! Return boundary condition at this direction index
  Constants::bc_type get_bc(const Constants::dir_type &direction) const {
    return bc[direction];
  }

  uint32_t get_structured_cells(void) const { return structured_cells; }

  bool get_random_problem(void) const { return random_problem; }

  uint64_t get_num_realizations(void) const { return num_realizations; }

  uint64_t get_realization_print(void) const {return realization_print; }

  uint32_t get_num_materials(void) const { return num_materials; }

  double get_problem_dist(void) const { return problem_dist; }

private:
  MPI_Datatype MPI_Region;
  // generator for random geometry profiles
  RNG *g_rng = new RNG();
  void seed_g_rng(int geometry_seed) { g_rng->set_seed(geometry_seed); }

  // random geometry data
  uint64_t num_realizations, realization_print;
  uint32_t num_materials;
  double column_size, problem_dist;
  uint32_t sublayer_cells, structured_cells, piecewise_segments;
  std::string chord_model;
  std::vector<std::string> mat_quad_minmax;
  std::vector<int> mat_id;
  std::vector<double> mat_chord_start, mat_chord_end, mat_quad_mult, mat_opacA, mat_opacB, mat_opacC, mat_opacS, mat_T_e, mat_T_r, mat_dens, mat_CV;

  // flags
  bool write_silo; //!< Dump SILO output files
  bool random_problem; //!< Makes use of random spatial region

  Constants::bc_type bc[6]; //!< Boundary condition array

  // timing
  double tStart;  //!< Starting time
  double dt;      //!< Timestep size
  double tFinish; //!< Finish time
  double tMult;   //!< Timestep multiplier
  double dtMax;   //!< Maximum timestep size

  // material
  std::vector<Region> regions; //!< Vector of regions in the problem

  //! Maps unique key to user set ID for a region
  std::map<uint32_t, uint32_t> region_map;

  //! Maps user set region ID to the index in the regions vector
  std::map<uint32_t, uint32_t> region_ID_to_index;

  // source
  double T_source; //!< Temperature of source

  // Monte Carlo parameters
  uint64_t n_photons; //!< Photons to source each timestep
  uint32_t seed;      //!< Random number seed

  // Method parameters
  bool use_tilt;  //!< Use tilting for emission sampling
  bool use_comb;  //!< Comb census photons
  bool use_strat; //!< Use strafifed sampling

  // Debug parameters
  uint32_t output_freq; //!< How often to print temperature information
  bool print_verbose;   //!< Verbose printing flag
  bool print_mesh_info; //!< Mesh information printing flag

  // detailed mesh specifications
  uint32_t n_divisions;            //!< Number of divisions in the mesh
  std::vector<double> x_start;     //!< x starting positions for each division
  std::vector<double> x_end;       //!< x ending positions for each division
  std::vector<double> y_start;     //!< y starting positions for each division
  std::vector<double> y_end;       //!< y ending positions for each division
  std::vector<double> z_start;     //!< z starting positions for each division
  std::vector<double> z_end;       //!< z ending positions for each division
  std::vector<uint32_t> n_x_cells; //!< Number of x cells in each division
  std::vector<uint32_t> n_y_cells; //!< Number of y cells in each division
  std::vector<uint32_t> n_z_cells; //!< Number of z cells in each division
  uint32_t n_global_x_cells; //!< Total number of x cells over all divisions
  uint32_t n_global_y_cells; //!< Total number of y cells over all divisions
  uint32_t n_global_z_cells; //!< Total number of z cells over all divisions

  // arrays for silo
  std::vector<float>
      silo_x; //!< x positions for SILO arrays (i + nx*j + nx*ny*k)
  std::vector<float>
      silo_y; //!< y positions for SILO arrays (i + nx*j + nx*ny*k)
  std::vector<float>
      silo_z; //!< z positions for SILO arrays (i + nx*j + nx*ny*k)
};

#endif // input_h_
//---------------------------------------------------------------------------//
// end of input.h
//---------------------------------------------------------------------------//
