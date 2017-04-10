#include "SolverProfiler.h"

namespace solver {
  template <typename T>
  SolverProfiler<T>::SolverProfiler(Solver<T> &solver): solver(solver) {}

  template <typename T>
  SolverProfiler<T>::~SolverProfiler() {}

  template <typename T>
  bool SolverProfiler<T>::add_constraint(const std::vector<T> constr) {
    return solver.add_constraint(constr);
  }

  template <typename T>
  void SolverProfiler<T>::set_bounds(const int idx, const T lower, const T upper) {
    return solver.set_bounds(idx, lower, upper);
  }

  template <typename T>
  std::vector<T> SolverProfiler<T>::solution() const {
    return solver.solution();
  }

  template <typename T>
  int SolverProfiler<T>::num_problem_vars() const {
    return solver.num_problem_vars();
  }

  template <typename T>
  int SolverProfiler<T>::num_additional_vars() const {
    return solver.num_additional_vars();
  }

  template <typename T>
  int SolverProfiler<T>::num_vars() const {
    return solver.num_vars();
  }

  template <typename T>
  int SolverProfiler<T>::num_constraints() const {
    return solver.num_constraints();
  }

  template <typename T>
  void SolverProfiler<T>::print_tableau() const {
    return solver.print_tableau();
  }

  template <typename T>
  void SolverProfiler<T>::print_variables() const {
    return solver.print_variables();
  }

  template <typename T>
  int SolverProfiler<T>::get_step_count() const {
    return solver.get_step_count();
  }

  template <typename T>
  bool SolverProfiler<T>::solve() {
    double time = -cpu_second();
    bool result = Solver<T>::solve();
    time += cpu_second();
    time_solve += time;
    return result;
  }

  template <typename T>
  bool SolverProfiler<T>::check_bounds(int &broken_idx) {
    double time = -cpu_second();
    bool result = solver.check_bounds(broken_idx);
    time += cpu_second();
    time_check_bounds += time;
    return result;
  }

  template <typename T>
  bool SolverProfiler<T>::find_suitable(const int broken_idx, int &suitable_idx) {
    double time = -cpu_second();
    bool result = solver.find_suitable(broken_idx, suitable_idx);
    time += cpu_second();
    time_find_suitable += time;
    return result;
  }

  template <typename T>
  void SolverProfiler<T>::pivot(const int broken_idx, const int suitable_idx) {
    double time = -cpu_second();
    solver.pivot(broken_idx, suitable_idx);
    time += cpu_second();
    time_pivot += time;
  }

  template <typename T>
  void SolverProfiler<T>::update_assignment() {
    double time = -cpu_second();
    solver.update_assignment();
    time += cpu_second();
    time_update_assignment += time;
  }
}  // namespace solver
