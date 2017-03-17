#ifndef SOLVER_H_
#define SOLVER_H_

#include "AbstractSolver.h"

namespace solver {
  template <typename T>
  class Solver : public AbstractSolver<T> {
    friend class SolverProfiler;
   public:
    // Factory method
    static Solver* create(SolverType type, const int num_vars, const int max_num_constrs);
    virtual void print_tableau() const = 0;
    virtual void print_variables() const = 0;
    virtual inline int get_step_count() const override {
      return steps_;
    }

   protected:
    virtual inline void incr_step_count() {
      steps_++;
    }
    // Converts a variable index into a string representation.
    inline std::string var2str(const int idx) const {
      const int ncols = this->num_problem_vars();
      if (idx < ncols) {
        return "x" + std::to_string(idx);
      } else {
        return "s" + std::to_string(idx - ncols);
      }
    }

   private:
    int steps_ = 0;
  };
}  // namespace solver

#endif /* SOLVER_H_ */
