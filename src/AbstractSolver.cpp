#include "AbstractSolver.h"

template <typename T>
solver::AbstractSolver<T>::~AbstractSolver() {}

template <typename T>
bool solver::AbstractSolver<T>::solve() {
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
      std::cout << get_step_count() << " steps" << std::endl;
    }
    double check_time = time + cpu_second();
    if (check_time >= 100.0) {
      printf("Bailing at %.2f seconds...\n", check_time);
      return false;
    }
#ifdef DEBUG
    printf("<Step %d>\n", get_step_count());
    printf("#### Variables ####\n");
    print_variables();
    printf("#### Tableau ####\n");
    print_tableau();
    printf("\n");
#endif
  }
  return true;
}

template class solver::AbstractSolver<float>;
template class solver::AbstractSolver<solver::Fraction>;
