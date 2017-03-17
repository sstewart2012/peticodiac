#ifndef ABSTRACTSOLVER_H_
#define ABSTRACTSOLVER_H_

#include <string>
#include <map>
#include <set>
#include <vector>
#include <cassert>
#include "timing.h"
#include "fraction.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif /* CUDA_ENABLED */

#define PRINT_STEP_FREQ 1000
#define OFFSET(row, col, ncols) (row * ncols + col)
#define NO_BOUND -1
#define EPSILON 0.000001
#define NONBASIC_FLAG 0
#define BASIC_FLAG 1
#define NONE_FOUND -1
#define IS_INCREASABLE(low, upp, ass) (upp == NO_BOUND || ass < upp)
#define IS_DECREASABLE(low, upp, ass) (low == NO_BOUND || ass > low)

namespace solver {
  enum SolverType {
    CPU_EAGER,
    CPU_LAZY,
    CUDA
  };

  template <typename T>
  class AbstractSolver {
   public:
    virtual ~AbstractSolver();
    virtual bool add_constraint(const std::vector<T> constr) = 0;
    virtual void set_bounds(const int idx, const T lower, const T upper) = 0;
    virtual std::vector<T> solution() const = 0;
    virtual bool solve();
    virtual int num_problem_vars() const = 0;
    virtual int num_additional_vars() const = 0;
    virtual int num_vars() const = 0;
    virtual int num_constraints() const = 0;
    virtual void print_tableau() const = 0;
    virtual void print_variables() const = 0;
    virtual int get_step_count() const = 0;

   protected:
    virtual bool check_bounds(int &broken_idx) = 0;
    virtual bool find_suitable(const int broken_idx, int &suitable_idx) = 0;
    virtual void pivot(const int broken_idx, const int suitable_idx) = 0;
    virtual void update_assignment() = 0;
  };
}  // namespace solver

#endif /* ABSTRACTSOLVER_H_ */
