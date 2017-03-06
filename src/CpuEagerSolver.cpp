#include "CpuEagerSolver.h"

solver::CpuEagerSolver::CpuEagerSolver(const int num_vars, const int max_num_constrs)
    : CpuSolver(num_vars, max_num_constrs) {}

solver::CpuEagerSolver::~CpuEagerSolver() {}

void solver::CpuEagerSolver::update_assignment() {
  incr_step_count();

  int i, j;
  #pragma omp parallel for private(i, j)
  for (i = 0; i < nrows_; ++i) {
    float accum = 0.0f;
    for (j = 0; j < ncols_; ++j) {
      accum += tableau_[i * ncols_ + j] * assigns_[col_to_var_[j]];
    }
    assigns_[row_to_var_[i]] = accum;
  }
}
