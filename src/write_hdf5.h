#ifndef write_hdf5_h_
#define write_hdf5_h_

#include <sstream>
#include <string>
#include <vector>

#include <iomanip>
#include <hdf5.h>

#include "config.h"
#include "constants.h"
#include "imc_state.h"

//! All ranks perform reductions to produce global arrays on rank zero
// writes the HDF5 file for visualization

void check_status(herr_t status) {
  if (status != 0) {
    std::cout << "Error code " << status << " in handling of HDF5 output file" << std::endl;
  }
}

void write_hdf5(const Mesh &mesh, const double &arg_time, const uint32_t &step, const double &r_transport_time, const double &r_mpi_time, const int &rank, const int &n_rank, bool rep_flag = true) {
#ifndef VIZ_LIBRARIES_FOUND
  using Constants::ELEMENT;
  using Constants::X_NEG;
  using Constants::X_POS;
  using Constants::Y_NEG;
  using Constants::Y_POS;
  using std::string;
  using std::stringstream;
  using std::vector;

  // need a non-const double to pass
  double time = arg_time;

  // generate a name for this hdf5 file
  stringstream ss;
  ss.setf(std::ios::showpoint);
  ss << std::setprecision(3);
  ss << "output_" << step << ".h5";
  string file = ss.str();

  int nx = mesh.get_global_n_x_faces();
  int ny = mesh.get_global_n_y_faces();
  int nz = mesh.get_global_n_z_faces();

  // set number of dimensions
  int ndims;
  // use a 1D mesh for only x cells
  if (ny == 2 && nz == 2)
    ndims = 1;
  // use a 2D mesh for one z cell (2 faces)
  else if (nz == 2)
    ndims = 2;
  // otherwise use 3D mesh for 3 or more z faces
  else
    ndims = 3;

  // generate title of plot
  stringstream tt;
  tt.setf(std::ios::showpoint);
  tt << std::setprecision(3);
  if (ndims == 2)
    tt << "2D rectangular mesh, t = " << time << " (sh)";
  else
    tt << "3D rectangular mesh, t = " << time << " (sh)";
  string title = tt.str();

  // get total cells for MPI all_reduce calls
  uint32_t n_xyz_cells;
  if (ndims == 1)
    n_xyz_cells = nx - 1;
  else if (ndims == 2)
    n_xyz_cells = (nx - 1) * (ny - 1);
  else
    n_xyz_cells = (nx - 1) * (ny - 1) * (nz - 1);

  // make vectors of data for plotting
  vector<int> region_data(n_xyz_cells, -1);
  vector<double> T_e(n_xyz_cells, 0.0);
  vector<double> T_r(n_xyz_cells, 0.0);
  vector<double> transport_time(n_xyz_cells, 0.0);
  vector<double> mpi_time(n_xyz_cells, 0.0);
  vector<int> grip_ID(n_xyz_cells, 0);

  // get rank data, map values from global ID to HDF5 ID
  uint32_t n_local = mesh.get_n_local_cells();
  Cell cell;
  uint32_t g_index, hdf5_index;
  for (uint32_t i = 0; i < n_local; i++) {
    cell = mesh.get_cell(i);
    g_index = cell.get_ID();
    hdf5_index = cell.get_silo_index();
    region_data[hdf5_index] = mesh.get_cell(i).get_region_ID();
    // set hdf5 plot variables
    T_e[hdf5_index] = cell.get_T_e();
    T_r[hdf5_index] = mesh.get_T_r(i);
    transport_time[hdf5_index] = r_transport_time;
    mpi_time[hdf5_index] = r_mpi_time;
    grip_ID[hdf5_index] = cell.get_grip_ID();
  }

  // don't reduce these quantities in replicated mode
  if (!rep_flag) {
    // reduce to get rank of each cell across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &region_data[0], n_xyz_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // reduce to get T_e across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &T_e[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // reduce to get T_r across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &T_r[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // reduce to get transport runtime from all ranks
    MPI_Allreduce(MPI_IN_PLACE, &transport_time[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // reduce to get mpi time from all ranks
    MPI_Allreduce(MPI_IN_PLACE, &mpi_time[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // reduce to get grip_ID across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &grip_ID[0], n_xyz_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  // First rank writes the HDF5 file
  if (rank == 0) {
    // write the global mesh
    auto x_vec = mesh.get_silo_x();
    auto y_vec = mesh.get_silo_y();
    auto z_vec = mesh.get_silo_z();
    float *x = &x_vec[0];
    float *y = &y_vec[0];
    float *z = &z_vec[0];

    hsize_t *dims;
    float **coords;
    hsize_t *cell_dims;
    uint32_t n_xyz_cells;

    // do 1D write
    if (ndims == 1) {
      dims = new hsize_t[1];
      dims[0] = nx;
      coords = new float *[1];
      coords[0] = x;
      cell_dims = new hsize_t[1];
      cell_dims[0] = nx - 1;
    }
    // do 2D write
    if (ndims == 2) {
      dims = new hsize_t[2];
      dims[0] = nx;
      dims[1] = ny;
      coords = new float *[2];
      coords[0] = x;
      coords[1] = y;
      cell_dims = new hsize_t[2];
      cell_dims[0] = nx - 1;
      cell_dims[1] = ny - 1;
    }
    // do 3D write
    else {
      dims = new hsize_t[3];
      dims[0] = nx;
      dims[1] = ny;
      dims[2] = nz;
      coords = new float *[3];
      coords[0] = x;
      coords[1] = y;
      coords[2] = z;
      cell_dims = new hsize_t[3];
      cell_dims[0] = nx - 1;
      cell_dims[1] = ny - 1;
      cell_dims[2] = nz - 1;
    }

    // create HDF5 file for this mesh
    herr_t status;

    hid_t file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // make the correct potion list for 2D and 3D meshes
    hid_t x_dataspace = H5Screate_simple(1, &dims[0], NULL);
    hid_t x_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    hid_t x_dataset = H5Dcreate(file_id, "x", x_datatype, x_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(x_dataset, x_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords[0]);
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
    if (ndims > 1) {
      hid_t y_dataspace = H5Screate_simple(1, &dims[1], NULL);
      hid_t y_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      hid_t y_dataset = H5Dcreate(file_id, "y", y_datatype, y_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(y_dataset, y_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords[1]);
      check_status(status);
      status = H5Tclose(y_datatype);
      check_status(status);
      hid_t y_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(y_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t y_unit_dims[1] = {1};
      hid_t y_unit_dataspace = H5Screate_simple(1, y_unit_dims, NULL);
      hid_t y_unit = H5Acreate(y_dataset, "units", y_unit_type, y_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *y_unit_str = "cm";
      status = H5Awrite(y_unit, y_unit_type, &y_unit_str);
      check_status(status);
      status = H5Sclose(y_unit_dataspace);
      check_status(status);
      status = H5Aclose(y_unit);
      check_status(status);
      status = H5Tclose(y_unit_type);
      check_status(status);
      status = H5Dclose(y_dataset);
      check_status(status);
      status = H5Sclose(y_dataspace);
      check_status(status);
    }
    if (ndims > 2) {
      hid_t z_dataspace = H5Screate_simple(1, &dims[2], NULL);
      hid_t z_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
      hid_t z_dataset = H5Dcreate(file_id, "z", z_datatype, z_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(z_dataset, z_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords[1]);
      check_status(status);
      status = H5Tclose(z_datatype);
      check_status(status);
      hid_t z_unit_type = H5Tcopy(H5T_C_S1);
      status = H5Tset_size(z_unit_type, H5T_VARIABLE);
      check_status(status);
      hsize_t z_unit_dims[1] = {1};
      hid_t z_unit_dataspace = H5Screate_simple(1, z_unit_dims, NULL);
      hid_t z_unit = H5Acreate(z_dataset, "units", z_unit_type, z_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
      char const *z_unit_str = "cm";
      status = H5Awrite(z_unit, z_unit_type, &z_unit_str);
      check_status(status);
      status = H5Sclose(z_unit_dataspace);
      check_status(status);
      status = H5Aclose(z_unit);
      check_status(status);
      status = H5Tclose(z_unit_type);
      check_status(status);
      status = H5Dclose(z_dataset);
      check_status(status);
      status = H5Sclose(z_dataspace);
      check_status(status);
    }

    // write rank IDs xy to 1D array
    /*
    std::vector<int> region_ids = region_data;
    std::sort(region_ids.begin(), region_ids.end());
    auto last = std::unique(region_ids.begin(), region_ids.end());
    region_ids.erase(last, region_ids.end());
    */

    hid_t reg_id_dataspace = H5Screate_simple(ndims, cell_dims, NULL);
    hid_t reg_id_datatype = H5Tcopy(H5T_NATIVE_INT);
    hid_t reg_id_dataset = H5Dcreate(file_id, "region_ID", reg_id_datatype, reg_id_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(reg_id_dataset, reg_id_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &region_data[0]);
    check_status(status);
    status = H5Tclose(reg_id_datatype);
    check_status(status);
    status = H5Dclose(reg_id_dataset);
    check_status(status);
    status = H5Sclose(reg_id_dataspace);
    check_status(status);

    // write the material temperature scalar field
    hid_t t_e_dataspace = H5Screate_simple(ndims, cell_dims, NULL);
    hid_t t_e_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    hid_t t_e_dataset = H5Dcreate(file_id, "t_e", t_e_datatype, t_e_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(t_e_dataset, t_e_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &T_e[0]);
    check_status(status);
    status = H5Tclose(t_e_datatype);
    check_status(status);
    hid_t t_e_unit_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(t_e_unit_type, H5T_VARIABLE);
    check_status(status);
    hsize_t t_e_unit_dims[1] = {1};
    hid_t t_e_unit_dataspace = H5Screate_simple(1, t_e_unit_dims, NULL);
    hid_t t_e_unit = H5Acreate(t_e_dataset, "units", t_e_unit_type, t_e_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    char const *t_e_unit_str = "keV";
    status = H5Awrite(t_e_unit, t_e_unit_type, &t_e_unit_str);
    check_status(status);
    status = H5Sclose(t_e_unit_dataspace);
    check_status(status);
    status = H5Aclose(t_e_unit);
    check_status(status);
    status = H5Tclose(t_e_unit_type);
    check_status(status);
    status = H5Dclose(t_e_dataset);
    check_status(status);
    status = H5Sclose(t_e_dataspace);
    check_status(status);

    // write the radiation temperature scalar field
    hid_t t_r_dataspace = H5Screate_simple(ndims, cell_dims, NULL);
    hid_t t_r_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    hid_t t_r_dataset = H5Dcreate(file_id, "t_r", t_r_datatype, t_r_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(t_r_dataset, t_r_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &T_r[0]);
    check_status(status);
    status = H5Tclose(t_r_datatype);
    check_status(status);
    hid_t t_r_unit_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(t_r_unit_type, H5T_VARIABLE);
    check_status(status);
    hsize_t t_r_unit_dims[1] = {1};
    hid_t t_r_unit_dataspace = H5Screate_simple(1, t_r_unit_dims, NULL);
    hid_t t_r_unit = H5Acreate(t_r_dataset, "units", t_r_unit_type, t_r_unit_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    char const *t_r_unit_str = "keV";
    status = H5Awrite(t_r_unit, t_r_unit_type, &t_r_unit_str);
    check_status(status);
    status = H5Sclose(t_r_unit_dataspace);
    check_status(status);
    status = H5Aclose(t_r_unit);
    check_status(status);
    status = H5Tclose(t_r_unit_type);
    check_status(status);
    status = H5Dclose(t_r_dataset);
    check_status(status);
    status = H5Sclose(t_r_dataspace);
    check_status(status);

    // free data
    delete[] cell_dims;
    delete[] coords;
    delete[] dims;

    // close file
    status = H5Fclose(file_id);
    check_status(status);
  } // end rank==0
#endif
}

#endif // write_hdf5_h_
