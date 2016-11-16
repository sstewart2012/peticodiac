#include "AbstractSolver.h"

namespace solver {

// Implementation of factory method
Solver* Solver::create(SolverType type, const int num_vars, const int max_num_constrs) {
  if (type == SolverType::CPU_LAZY) {
    printf("CPU_LAZY\n");
    return new CpuSolver(num_vars, max_num_constrs);
  } else if (type == SolverType::CPU_EAGER) {
    printf("CPU_EAGER\n");
    return new CpuEagerSolver(num_vars, max_num_constrs);
  } else if (type == SolverType::CUDA) {
#ifdef CUDA_ENABLED
    printf("CUDA\n");
    return new CudaSolver(num_vars, max_num_constrs, 0);
#else
    printf("CUDA not supported\n");
#endif
  } else {
    return nullptr;
  }
}

} // namespace solver
