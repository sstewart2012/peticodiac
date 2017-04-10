#include "CpuEagerSolver.h"

template <typename T>
solver::CpuEagerSolver<T>::CpuEagerSolver(const int num_vars, const int max_num_constrs)
    : CpuSolver<T>(num_vars, max_num_constrs) {}

template <typename T>
solver::CpuEagerSolver<T>::~CpuEagerSolver() {}

template <typename T>
void solver::CpuEagerSolver<T>::update_assignment() {
  this->incr_step_count();

  int i, j;
  #pragma omp parallel for private(i, j)
  for (i = 0; i < this->nrows_; ++i) {
    float accum = 0.0f;
    for (j = 0; j < this->ncols_; ++j) {
      accum += this->tableau_[i * this->ncols_ + j] * this->assigns_[this->col_to_var_[j]];
    }
    this->assigns_[this->row_to_var_[i]] = accum;
  }
}

template class solver::CpuEagerSolver<float>;
