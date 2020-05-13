#ifndef random_overlay_h_
#define random_overlay_h_

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <hdf5.h>

#include "running_statistics.h"
#include "constants.h"
#include "imc_state.h"

class Random_Overlay {
public:
  //! constructor for binary material system
  Random_Overlay(int num_cells_in, double problem_dist_in, int num_materials_in) : num_cells(num_cells_in), problem_dist(problem_dist_in), num_materials(num_materials_in) {
    structured_delta = problem_dist / num_cells;
    structured_T_e.resize(num_materials, std::vector<Running_Statistics>(num_cells));
    structured_T_r.resize(num_materials, std::vector<Running_Statistics>(num_cells));
  }

  void process_output(Mesh &mesh, const int &rank, const int &n_rank, bool rep_flag = true) {
    int nx = mesh.get_global_n_x_faces();

    int n_x_cells = (nx - 1);

    std::vector<int> region_data(n_x_cells, -1);
    std::vector<double> T_e(n_x_cells, 0.0);
    std::vector<double> T_r(n_x_cells, 0.0);

    uint32_t n_local = mesh.get_n_local_cells();
    Cell cell;
    uint32_t g_index, buffer_index;
    for (uint32_t i = 0; i < n_local; i++) {
      cell = mesh.get_cell(i);
      g_index = cell.get_ID();
      buffer_index = cell.get_silo_index();
      region_data[buffer_index] = mesh.get_cell(i).get_region_ID();
      T_e[buffer_index] = cell.get_T_e();
      T_r[buffer_index] = mesh.get_T_r(i);
    }

    // don't reduce these quantities in replicated mode
    if (!rep_flag) {
      // reduce to get rank of each cell across all ranks
      MPI_Allreduce(MPI_IN_PLACE, &region_data[0], n_x_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // reduce to get T_e across all ranks
      MPI_Allreduce(MPI_IN_PLACE, &T_e[0], n_x_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // reduce to get T_r across all ranks
      MPI_Allreduce(MPI_IN_PLACE, &T_r[0], n_x_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // First rank does the mesh-mapping calculation
    if (rank == 0) {
      auto x_vec = mesh.get_silo_x();
      float *x = &x_vec[0];

      for (int k = 0; k < num_materials; k++) {
        // Assign counter to negative one due to pre-fetch increment
        int counter = -1;

        // Assign tallies
        double distance_tally = 0.0;  // cm
        double unstructured_distance_tally = 0.0;  // cm
        double structured_distance_tally = 0.0;  // cm
        double leftover_distance = 0.0;  // cm
        double m_switch = 0.0;
        double delta = 0.0;  // cm

        // Loop for mapping the unstructured to the structured mesh
        for (int i = 0; i < num_cells; i++) {
          // Start from the border with a zero weight for the current cell
          bool distance_overlap = false;
          double t_e_weight_tally = 0.0;  // unit-cm
          double t_r_weight_tally = 0.0;  // unit-cm

          // Increment structured distance tally
          structured_distance_tally += structured_delta;  // cm

          // Carry leftover distance
          if (leftover_distance > 0.0) {
            // If leftover distance is still over-reaching tally boundaries
            if ((distance_tally + leftover_distance) >= structured_distance_tally) {
              delta = structured_delta;  // cm
              leftover_distance = unstructured_distance_tally - structured_distance_tally;  // cm
              distance_overlap = true;
            } else {
              delta = leftover_distance;  // cm
              leftover_distance = 0.0;  // cm
            }
            t_e_weight_tally += delta * T_e[counter] * m_switch;  // unit-cm
            t_r_weight_tally += delta * T_r[counter] * m_switch;  // unit-cm
            distance_tally += delta;  // cm
          }

          while ((!distance_overlap) && (counter < n_x_cells)) {
            // Increment counter (unstructured index)
            counter += 1;
            // Increment unstructured distance tally
            double unstructured_delta = x[counter + 1] - x[counter];
            unstructured_distance_tally += unstructured_delta;  // cm

            // Material number for calculations
            // Tally switch for each material
            if (region_data[counter] == k)
              m_switch = 1.0;
            else
              m_switch = 0.0;

            // Check for boundary overlap
            if ((unstructured_distance_tally >= structured_distance_tally) || (counter == n_x_cells)) {
              delta = structured_distance_tally - distance_tally;  // cm
              leftover_distance = unstructured_distance_tally - structured_distance_tally;  // cm
              distance_overlap = true;
            } else {
              delta = unstructured_delta;  // cm
            }

            // Increment the known distance tally
            distance_tally += delta;  // cm

            // Apply linear weighted tally
            t_e_weight_tally += delta * T_e[counter] * m_switch;  // unit-cm
            t_r_weight_tally += delta * T_r[counter] * m_switch;  // unit-cm
          }  // Unstructured loop

          // Average the results, or just append if no results previously
          if (t_e_weight_tally != 0.0) {
            structured_T_e.at(k).at(i).push(t_e_weight_tally / structured_delta);  // unit
          }
          if (t_r_weight_tally != 0.0) {
            structured_T_r.at(k).at(i).push(t_r_weight_tally / structured_delta);  // unit
          }
        }  // Structured loop
      }  // Material loop
    }  // Rank check
  }

  void write_hdf5(const int &rank, uint32_t step_num) {
    // First rank writes the HDF5 file
    if (rank == 0) {
      std::stringstream file_name_ss;
      file_name_ss.setf(std::ios::showpoint);
      file_name_ss << std::setprecision(3);
      file_name_ss << "random_output_" << step_num << ".h5";
      std::string file_name = file_name_ss.str();

      std::cout << "Writing: " << file_name << std::endl;

      // Spatial vector
      std::vector<double> structured_space(num_cells + 1, 0.0);
      for (int i = 0; i <= num_cells; i++) {
        structured_space[i] = structured_delta * i;
      }

      // Independent material vectors
      std::vector<double> material_0_T_e(num_cells, 0.0);
      std::vector<double> material_1_T_e(num_cells, 0.0);
      std::vector<double> material_0_T_r(num_cells, 0.0);
      std::vector<double> material_1_T_r(num_cells, 0.0);
      for (int i = 0; i < num_cells; i++) {
        material_0_T_e[i] = structured_T_e.at(0).at(i).mean();
        material_1_T_e[i] = structured_T_e.at(1).at(i).mean();
        material_0_T_r[i] = structured_T_r.at(0).at(i).mean();
        material_1_T_r[i] = structured_T_r.at(1).at(i).mean();
      }

      // Create HDF5 file
      herr_t status;
      hsize_t spatial_data_size = num_cells + 1;
      hsize_t data_size = num_cells;
      hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write spatial data
      hid_t x_dataspace = H5Screate_simple(1, &spatial_data_size, NULL);
      hid_t x_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      hid_t x_dataset = H5Dcreate(file_id, "x", x_datatype, x_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(x_dataset, x_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &structured_space[0]);
      check_status(status);
      status = H5Tclose(x_datatype);
      check_status(status);
      hid_t x_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(x_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t x_unit_dims[1] = {1};
      hid_t x_unit_dataspace = H5Screate_simple(1, x_unit_dims, NULL);
      hid_t x_unit = H5Acreate(x_dataset, "units", x_unit_type, x_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *x_unit_str = "cm";
      status = H5Awrite(x_unit, x_unit_type, &x_unit_str);
      check_status(status);
      status = H5Sclose(x_unit_dataspace);
      check_status(status);
      status = H5Aclose(x_unit);
      check_status(status);
      status = H5Tclose(x_unit_type);
      check_status(status);
      status = H5Dclose(x_dataset);
      check_status(status);
      status = H5Sclose(x_dataspace);
      check_status(status);

      // Write material 0 T_e
      hid_t m0_t_e_dataspace = H5Screate_simple(1, &data_size, NULL);
      hid_t m0_t_e_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      hid_t m0_t_e_dataset = H5Dcreate(file_id, "m0_t_e", m0_t_e_datatype, m0_t_e_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(m0_t_e_dataset, m0_t_e_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &material_0_T_e[0]);
      check_status(status);
      status = H5Tclose(m0_t_e_datatype);
      check_status(status);
      hid_t m0_t_e_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(m0_t_e_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t m0_t_e_unit_dims[1] = {1};
      hid_t m0_t_e_unit_dataspace = H5Screate_simple(1, m0_t_e_unit_dims, NULL);
      hid_t m0_t_e_unit = H5Acreate(m0_t_e_dataset, "units", m0_t_e_unit_type, m0_t_e_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *m0_t_e_unit_str = "keV";
      status = H5Awrite(m0_t_e_unit, m0_t_e_unit_type, &m0_t_e_unit_str);
      check_status(status);
      status = H5Sclose(m0_t_e_unit_dataspace);
      check_status(status);
      status = H5Aclose(m0_t_e_unit);
      check_status(status);
      status = H5Tclose(m0_t_e_unit_type);
      check_status(status);
      status = H5Dclose(m0_t_e_dataset);
      check_status(status);
      status = H5Sclose(m0_t_e_dataspace);
      check_status(status);

      // Write material 1 T_e
      hid_t m1_t_e_dataspace = H5Screate_simple(1, &data_size, NULL);
      hid_t m1_t_e_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      hid_t m1_t_e_dataset = H5Dcreate(file_id, "m1_t_e", m1_t_e_datatype, m1_t_e_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(m1_t_e_dataset, m1_t_e_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &material_1_T_e[0]);
      check_status(status);
      status = H5Tclose(m1_t_e_datatype);
      check_status(status);
      hid_t m1_t_e_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(m1_t_e_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t m1_t_e_unit_dims[1] = {1};
      hid_t m1_t_e_unit_dataspace = H5Screate_simple(1, m1_t_e_unit_dims, NULL);
      hid_t m1_t_e_unit = H5Acreate(m1_t_e_dataset, "units", m1_t_e_unit_type, m1_t_e_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *m1_t_e_unit_str = "keV";
      status = H5Awrite(m1_t_e_unit, m1_t_e_unit_type, &m1_t_e_unit_str);
      check_status(status);
      status = H5Sclose(m1_t_e_unit_dataspace);
      check_status(status);
      status = H5Aclose(m1_t_e_unit);
      check_status(status);
      status = H5Tclose(m1_t_e_unit_type);
      check_status(status);
      status = H5Dclose(m1_t_e_dataset);
      check_status(status);
      status = H5Sclose(m1_t_e_dataspace);
      check_status(status);

      // Write material 0 T_r
      hid_t m0_t_r_dataspace = H5Screate_simple(1, &data_size, NULL);
      hid_t m0_t_r_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      hid_t m0_t_r_dataset = H5Dcreate(file_id, "m0_t_r", m0_t_r_datatype, m0_t_r_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(m0_t_r_dataset, m0_t_r_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &material_0_T_r[0]);
      check_status(status);
      status = H5Tclose(m0_t_r_datatype);
      check_status(status);
      hid_t m0_t_r_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(m0_t_r_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t m0_t_r_unit_dims[1] = {1};
      hid_t m0_t_r_unit_dataspace = H5Screate_simple(1, m0_t_r_unit_dims, NULL);
      hid_t m0_t_r_unit = H5Acreate(m0_t_r_dataset, "units", m0_t_r_unit_type, m0_t_r_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *m0_t_r_unit_str = "keV";
      status = H5Awrite(m0_t_r_unit, m0_t_r_unit_type, &m0_t_r_unit_str);
      check_status(status);
      status = H5Sclose(m0_t_r_unit_dataspace);
      check_status(status);
      status = H5Aclose(m0_t_r_unit);
      check_status(status);
      status = H5Tclose(m0_t_r_unit_type);
      check_status(status);
      status = H5Dclose(m0_t_r_dataset);
      check_status(status);
      status = H5Sclose(m0_t_r_dataspace);
      check_status(status);

      // Write material 1 T_r
      hid_t m1_t_r_dataspace = H5Screate_simple(1, &data_size, NULL);
      hid_t m1_t_r_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      hid_t m1_t_r_dataset = H5Dcreate(file_id, "m1_t_r", m1_t_r_datatype, m1_t_r_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(m1_t_r_dataset, m1_t_r_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &material_1_T_r[0]);
      check_status(status);
      status = H5Tclose(m1_t_r_datatype);
      check_status(status);
      hid_t m1_t_r_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(m1_t_r_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t m1_t_r_unit_dims[1] = {1};
      hid_t m1_t_r_unit_dataspace = H5Screate_simple(1, m1_t_r_unit_dims, NULL);
      hid_t m1_t_r_unit = H5Acreate(m1_t_r_dataset, "units", m1_t_r_unit_type, m1_t_r_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *m1_t_r_unit_str = "keV";
      status = H5Awrite(m1_t_r_unit, m1_t_r_unit_type, &m1_t_r_unit_str);
      check_status(status);
      status = H5Sclose(m1_t_r_unit_dataspace);
      check_status(status);
      status = H5Aclose(m1_t_r_unit);
      check_status(status);
      status = H5Tclose(m1_t_r_unit_type);
      check_status(status);
      status = H5Dclose(m1_t_r_dataset);
      check_status(status);
      status = H5Sclose(m1_t_r_dataspace);
      check_status(status);

      // Close file
      status = H5Fclose(file_id);
      check_status(status);
    }
  }

  //! destructor
  ~Random_Overlay() { }

private:
  int num_materials;
  int num_cells;
  double problem_dist;
  double structured_delta;
  std::vector<std::vector<Running_Statistics>> structured_T_e, structured_T_r;

  void check_status(herr_t status) {
    if (status != 0) {
      std::cout << "Error code " << status << " in handling of HDF5 output file" << std::endl;
    }
  }
};

#endif
