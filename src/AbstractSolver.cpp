#include "AbstractSolver.h"

solver::AbstractSolver::~AbstractSolver() {}

bool solver::AbstractSolver::solve() {
  double time = -cpu_second();
  int broken_idx = -1;
  int suitable_idx = -1;
  while (!check_bounds(broken_idx)) {
    if (!find_suitable(broken_idx, suitable_idx)) {
      return false;
    }
    pivot(broken_idx, suitable_idx);
    update_assignment();
    if (get_step_count() % 1000 == 0) {
      printf("%d steps\n", get_step_count());
    }
    double check_time = time + cpu_second();
    if (check_time >= 10.0) {
      printf("Bailing at %.2f seconds...\n", check_time);
      return false;
    }
#ifdef DEBUG
    printf("<Step %d>\n", get_step_count());
    print_variables();
    print_tableau();
#endif
  }
  return true;
}
