#include <iostream>
#include <cstdio>
#include <cmath>
#include <omp.h>

#include "AbstractSolver.h"
#include "random.h"
#include "timing.h"

using solver::Solver;
using solver::SolverType;
#ifdef PROFILER_ENABLED
using solver::SolverProfiler;
#endif

void print_solution(const Solver &s) {
  const std::vector<float> solution = s.solution();
  for (int i = 0; i < solution.size(); ++i) {
    printf("x%d=%.3f%s", i, solution[i], i + 1 < solution.size() ? " " : "\n");
  }
}

void test_random(Solver* const solver, const int num_vars, const int num_constrs) {
  for (int i = 0; i < num_constrs; ++i) {
    std::vector<float> constr = ::random_instance(num_vars, 0, 11);
    solver->add_constraint(constr);
    const float* const bounds = ::random_float(2, 1, 11);
    const float low = std::min(bounds[0], bounds[1]);
    const float upp = std::max(bounds[0], bounds[1]);
    if (low == upp) {
      solver->set_bounds(i + num_vars, low, upp + 1);
    } else {
      solver->set_bounds(i + num_vars, low, upp);
    }
  }

#ifdef DEBUG
  solver->print_variables();
  printf("\n");
  solver->print_tableau();
  printf("\n");
#endif

  double solver_time = -cpu_second();
  bool solver_val = solver->solve();
  solver_time += cpu_second();
  int solver_steps = solver->get_step_count();

  printf("CPU time              : %.3f seconds\n", solver_time);
  printf("CPU result            : %s\n", solver_val ? "SAT" : "UNSAT");
  printf("CPU steps             : %d\n", solver_steps);
  printf("CPU steps per second  : %.1f\n", solver_steps / solver_time);
  printf("\n");
}

void test_solver(const SolverType type, const int num_vars,
                 const int num_constrs) {
  Solver* solver = Solver::create(type, 2, 3);
  solver->add_constraint( { 1, 1 });
  solver->add_constraint( { 2, -1 });
  solver->add_constraint( { -1, 2 });
  solver->set_bounds(2, 2, NO_BOUND);
  solver->set_bounds(3, 0, NO_BOUND);
  solver->set_bounds(4, 1, NO_BOUND);

  double time = -cpu_second();
  bool val = solver->solve();
  time += cpu_second();

  if (val) {
    printf("SAT (%.3f seconds)\n", time);
    print_solution(*solver);
  } else {
    printf("UNSAT (%.3f seconds)\n", time);
  }
  delete solver;
}

int main(const int argc, const char** argv) {
  const int num_vars = atoi(argv[1]);
  const int num_constrs = atoi(argv[2]);
  const int solver_type = atoi(argv[3]);

  Solver *solver = nullptr;
  if (solver_type == 1)
    solver = Solver::create(SolverType::CPU_EAGER, num_vars, num_constrs);
  else if (solver_type == 2)
    solver = Solver::create(SolverType::CPU_LAZY, num_vars, num_constrs);
  else if (solver_type == 3)
    solver = Solver::create(SolverType::CUDA, num_vars, num_constrs);

  if (solver != nullptr) {
    test_random(solver, num_vars, num_constrs);
    delete solver;
  }

  exit(EXIT_SUCCESS);
}
