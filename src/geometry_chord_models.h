#ifndef geometry_chord_models_h_
#define geometry_chord_models_h_

#include <algorithm>

// Linear chord model
double linear_chord(double chord_initial, double chord_slope, double distance) {
  return chord_initial + chord_slope * distance;
}

// Quadratic chord model
double quad_chord(double param_a, double param_b, double param_c, double distance) {
  return param_a * pow(distance, 2) + param_b * distance + param_c;
}

// Supporting functions for quadratic chord model
// Midpoint location is multiplier-fold largest endpoint
double mid_value_max_func(double start_value, double end_value, double multiplier) {
  return std::max(multiplier * start_value, multiplier * end_value);
}

// Midpoint location is multiplier-fold smallest endpoint
double mid_value_min_func(double start_value, double end_value, double multiplier) {
  return std::min(multiplier * start_value, multiplier * end_value);
}

double calc_c(double start_value) {
  return start_value;
}

double calc_b(double delta_m, double delta_t, double end_dist) {
  return (2.0 * delta_m) / end_dist * (1.0 + sqrt(1.0 - delta_t / delta_m));
}

double calc_a(double delta_t, double param_b, double end_dist) {
  return (delta_t - param_b * end_dist) / pow(end_dist, 2);
}

// Piecewise constant chord functions
// Linear model
double piecewise_constant_linear_chord(double start_value, double end_value, int num_segments, double end_dist, double distance) {
  double slope = (end_value - start_value) / end_dist;
  double delta_dist = end_dist / static_cast<double>(num_segments);
  int bin_no = static_cast<int>(distance / delta_dist);
  if (distance >= end_dist)
    bin_no = num_segments - 1;
  double init_val = linear_chord(start_value, slope, delta_dist * bin_no);
  double end_val = linear_chord(start_value, slope, delta_dist * (bin_no + 1));
  return (init_val + end_val) / 2.0;
}

// Quadratic model
double piecewise_constant_quad_chord(double param_a, double param_b, double param_c, int num_segments, double end_dist, double distance) {
  double delta_dist = end_dist / static_cast<double>(num_segments);
  int bin_no = static_cast<int>(distance / delta_dist);
  if (distance >= end_dist)
    bin_no = num_segments - 1;
  double seg_init_distance = delta_dist * bin_no;
  double seg_end_distance = delta_dist * (bin_no + 1);
  // Preserve quadratic average
  return (param_a / 3.0) * (pow(seg_end_distance + seg_init_distance, 2) - seg_end_distance * seg_init_distance) + (param_b / 2.0) * (seg_end_distance + seg_init_distance) + param_c;
}

#endif
