#include <cmath>
#include <cstring>
#include "AbstractSolver.h"
#include <omp.h>

namespace solver {

void CpuEagerSolver::update_assignment() {
    incr_step_count();
#pragma omp parallel for
    for (int i = 0; i < nrows_; ++i) {
      float accum = 0.0f;
      for (int j = 0; j < ncols_; ++j) {
        accum += tableau_[i * ncols_ + j] * assigns_[col_to_var_[j]];
      }
      assigns_[row_to_var_[i]] = accum;
    }
  }

CpuSolver::CpuSolver(const int num_vars, const int max_num_constrs)
    : ncols_(num_vars),
      nrows_(max_num_constrs) {
  memset(tableau_, 0, nrows_ * ncols_ * sizeof(float));
  int i, j;
  for (i = 0; i < ncols_; ++i) {
    col_to_var_[i] = i;
    var_to_tableau_[i] = i;
    lower_[i] = 0.0f;
    upper_[i] = NO_BOUND;
    assigns_[i] = 0.0f;
    nonbasic_.insert(i);
    map_assigns_[i] = 0;
  }
  for (j = 0; j < nrows_; ++j, ++i) {
    row_to_var_[j] = i;
    var_to_tableau_[i] = j;
    lower_[i] = 0.0f;
    upper_[i] = 0.0f;
    assigns_[i] = 0.0f;
    basic_.insert(i);
    map_assigns_[i] = 0;
  }
}

CpuSolver::~CpuSolver() {
  delete[] tableau_;
  delete[] lower_;
  delete[] upper_;
  delete[] assigns_;
  delete[] row_to_var_;
  delete[] col_to_var_;
  delete[] var_to_tableau_;
}

bool CpuSolver::add_constraint(const std::vector<float> constr) {
#ifdef DEBUG
  printf("======== add constraint ========\n");
  printf("constr size = %u and ncols = %u\n", constr.size(), ncols_);
  printf("constr value = %f\n", constr[0]);
#endif
  if (constr.size() != ncols_)
    return false;
  std::memcpy(&tableau_[next_constr_idx_ * ncols_], &constr[0],
              constr.size() * sizeof(float));
  next_constr_idx_++;
  return true;
}

void CpuSolver::set_bounds(const int idx, const float lower,
                           const float upper) {
  lower_[idx] = lower;
  upper_[idx] = upper;
}

std::vector<float> CpuSolver::solution() const {
  std::vector<float> s;
  for (int i = 0; i < ncols_; ++i)
    s.push_back(assigns_[i]);
  return s;
}

void CpuSolver::print_tableau() const {
  for (auto it = basic_.begin(); it != basic_.end(); ++it) {
    int i = *it;
    printf("b%d %d:[%s] %.3f <= %.3f <= %.3f%s\n", var_to_tableau_[i], i,
           this->var2str(i).c_str(), lower_[i], assigns_[i], upper_[i],
           is_broken(i) ? " broken" : "");
  }
  for (auto it = nonbasic_.begin(); it != nonbasic_.end(); ++it) {
    int i = *it;
    printf("n%d %d:[%s] %.3f <= %.3f <= %.3f%s\n", var_to_tableau_[i], i,
           this->var2str(i).c_str(), lower_[i], assigns_[i], upper_[i],
           is_broken(i) ? " broken" : "");
  }
}

void CpuSolver::print_variables() const {
  for (int i = 0; i < nrows_; ++i) {
    for (int j = 0; j < ncols_; ++j)
      printf("%.3f%s", tableau_[i * ncols_ + j], j < ncols_ ? " " : "");
    printf("\n");
  }
}

bool CpuSolver::is_broken(const int idx) const {
  const float ass = get_assignment(idx);
  const float low = get_lower(idx);
  const float upp = get_upper(idx);
  if (fabs(ass - low) < EPSILON)  // "close enough" to lower bound
    return false;
  else if (fabs(ass - upp) < EPSILON)  // "close enough" to upper bound
    return false;
  else if (low != NO_BOUND && ass < low)
    return true;
  else if (upp != NO_BOUND && ass > upp)
    return true;
  else
    return false;
}

float CpuSolver::compute_assignment(const int idx) const {
  const int rowIdx = var_to_tableau_[idx];
  const float* const row = &tableau_[rowIdx * ncols_];
  float val = 0.0f;
#pragma omp parallel for
  for (int i = 0; i < ncols_; ++i) {
    val += row[i] * assigns_[col_to_var_[i]];
  }

  // Update the assignment
  const_cast<CpuSolver*>(this)->assigns_[idx] = val;
  const_cast<CpuSolver*>(this)->map_assigns_[idx] = get_step_count();

  return val;
}

bool CpuSolver::check_bounds(int &broken_idx) {
  for (auto it = basic_.begin(); it != basic_.end(); it++) {
    const int var = *it;
    if (is_broken(var)) {
      broken_idx = var;
      return false;
    }
  }
  return true;
}

bool CpuSolver::find_suitable(const int broken_idx, int &suitable_idx) {
  const float ass = assigns_[broken_idx];
  const float low = lower_[broken_idx];
  const bool increase = ass < low;
  const float delta = increase ? low - ass : ass - upper_[broken_idx];
  if (increase) {
    return find_suitable_increase(broken_idx, suitable_idx, delta);
  } else {
    return find_suitable_decrease(broken_idx, suitable_idx, delta);
  }
}

bool CpuSolver::find_suitable_increase(const int broken_idx, int &suitable_idx,
                                       const float delta) {
  for (auto it = nonbasic_.begin(); it != nonbasic_.end(); ++it) {
    const int var = *it;
    const float coeff = tableau_[var_to_tableau_[broken_idx] * ncols_
        + var_to_tableau_[var]];
    const float low = lower_[var];
    const float upp = upper_[var];
    const float ass = assigns_[var];
    if ((IS_INCREASABLE(low, upp, ass) && coeff > 0)
        || (IS_DECREASABLE(low, upp, ass) && coeff < 0)) {
      const float theta = delta / coeff;
      assigns_[var] += coeff < 0 ? -theta : theta;
      assigns_[broken_idx] += delta;
      suitable_idx = var;
      return true;
    }
  }
  return false;
}

bool CpuSolver::find_suitable_decrease(const int broken_idx, int &suitable_idx,
                                       const float delta) {
  for (auto it = nonbasic_.begin(); it != nonbasic_.end(); ++it) {
    const int var = *it;
    const float coeff = tableau_[var_to_tableau_[broken_idx] * ncols_
        + var_to_tableau_[var]];
    const float low = lower_[var];
    const float upp = upper_[var];
    const float ass = assigns_[var];
    if ((IS_INCREASABLE(low, upp, ass) && coeff < 0)
        || (IS_DECREASABLE(low, upp, ass) && coeff > 0)) {
      const float theta = delta / coeff;
      assigns_[var] -= coeff < 0 ? theta : -theta;
      assigns_[broken_idx] -= delta;
      suitable_idx = var;
      return true;
    }
  }
  return false;
}

void CpuSolver::pivot(const int broken_idx, const int suitable_idx) {
  const int pivot_row = var_to_tableau_[broken_idx];
  const int pivot_col = var_to_tableau_[suitable_idx];
  //printf("Pivot(%d,%d)\n", pivot_row, pivot_col);

  // Save the current pivot element (alpha)
  const int alpha_idx = OFFSET(pivot_row, pivot_col, ncols_);
  const float alpha = tableau_[alpha_idx];

  // Update the tableau
  pivot_update_inner(alpha, pivot_row, pivot_col);
  pivot_update_row(alpha, pivot_row);
  pivot_update_column(alpha, pivot_col);
  tableau_[alpha_idx] = 1.0f / alpha;

  // Swap the basic and nonbasic variables
  swap(pivot_row, pivot_col, broken_idx, suitable_idx);
}

void CpuSolver::swap(const int row, const int col, const int basic_idx,
                     const int nonbasic_idx) {
  col_to_var_[col] = basic_idx;
  row_to_var_[row] = nonbasic_idx;
  var_to_tableau_[basic_idx] = col;
  var_to_tableau_[nonbasic_idx] = row;
  basic_.erase(basic_idx);
  basic_.insert(nonbasic_idx);
  nonbasic_.erase(nonbasic_idx);
  nonbasic_.insert(basic_idx);
}

void CpuSolver::pivot_update_inner(const float alpha, const int row,
                                   const int col) {
#pragma omp parallel for
  for (int i = 0; i < nrows_; ++i) {
    if (i == row)
      continue;
    for (int j = 0; j < ncols_; ++j) {
      if (j == col)
        continue;
      const int delta_row_idx = i * ncols_;
      const int delta_idx = delta_row_idx + j;
      const float delta = tableau_[delta_idx];
      const float beta = tableau_[row * ncols_ + j];
      const float gamma = tableau_[delta_row_idx + col];
      tableau_[delta_idx] = delta - (beta * gamma) / alpha;
    }
  }
}

void CpuSolver::pivot_update_row(const float alpha, const int row) {
  float* beta = &tableau_[row * ncols_];
#pragma omp parallel for
  for (int i = 0; i < ncols_; ++i) {
    *beta = -(*beta) / alpha;
    beta++;
  }
}

void CpuSolver::pivot_update_column(const float alpha, const int col) {
  float* gamma = &tableau_[col];
#pragma omp parallel for
  for (int i = 0; i < nrows_; ++i) {
    *gamma /= alpha;
    gamma += ncols_;
  }
}

}  // namespace solver
