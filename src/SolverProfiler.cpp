#include "SolverProfiler.h"

namespace solver {
  SolverProfiler::SolverProfiler(Solver &solver): solver(solver) {}

  SolverProfiler::~SolverProfiler() {}

  bool SolverProfiler::add_constraint(const std::vector<float> constr) {
    return solver.add_constraint(constr);
  }

  void SolverProfiler::set_bounds(const int idx, const float lower, const float upper) {
    return solver.set_bounds(idx, lower, upper);
  }

  std::vector<float> SolverProfiler::solution() const {
    return solver.solution();
  }

  int SolverProfiler::num_problem_vars() const {
    return solver.num_problem_vars();
  }

  int SolverProfiler::num_additional_vars() const {
    return solver.num_additional_vars();
  }

  int SolverProfiler::num_vars() const {
    return solver.num_vars();
  }

  int SolverProfiler::num_constraints() const {
    return solver.num_constraints();
  }

  void SolverProfiler::print_tableau() const {
    return solver.print_tableau();
  }

  void SolverProfiler::print_variables() const {
    return solver.print_variables();
  }

  int SolverProfiler::get_step_count() const {
    return solver.get_step_count();
  }

  bool SolverProfiler::solve() {
    double time = -cpu_second();
    bool result = Solver::solve();
    time += cpu_second();
    time_solve += time;
    return result;
  }

  bool SolverProfiler::check_bounds(int &broken_idx) {
    double time = -cpu_second();
    bool result = solver.check_bounds(broken_idx);
    time += cpu_second();
    time_check_bounds += time;
    return result;
  }

  bool SolverProfiler::find_suitable(const int broken_idx, int &suitable_idx) {
    double time = -cpu_second();
    bool result = solver.find_suitable(broken_idx, suitable_idx);
    time += cpu_second();
    time_find_suitable += time;
    return result;
  }

  void SolverProfiler::pivot(const int broken_idx, const int suitable_idx) {
    double time = -cpu_second();
    solver.pivot(broken_idx, suitable_idx);
    time += cpu_second();
    time_pivot += time;
  }

  void SolverProfiler::update_assignment() {
    double time = -cpu_second();
    solver.update_assignment();
    time += cpu_second();
    time_update_assignment += time;
  }
}  // namespace solver
