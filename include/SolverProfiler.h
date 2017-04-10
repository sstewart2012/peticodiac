#ifndef SOLVERPROFILER_H_
#define SOLVERPROFILER_H_

#include "Solver.h"

namespace solver {
  template <typename T>
  class SolverProfiler : public Solver<T> {
  public:
    SolverProfiler(Solver<T> &solver);
    virtual ~SolverProfiler();

    virtual bool add_constraint(const std::vector<T> constr) override;
    virtual void set_bounds(const int idx, const T lower, const T upper) override;
    virtual std::vector<T> solution() const override;
    virtual int num_problem_vars() const override;
    virtual int num_additional_vars() const override;
    virtual int num_vars() const override;
    virtual int num_constraints() const override;
    virtual void print_tableau() const override;
    virtual void print_variables() const override;
    virtual int get_step_count() const override;
    virtual bool solve() override;

  protected:
    virtual bool check_bounds(int &broken_idx) override;
    virtual bool find_suitable(const int broken_idx, int &suitable_idx) override;
    virtual void pivot(const int broken_idx, const int suitable_idx) override;
    virtual void update_assignment() override;

   public:
    double time_check_bounds = 0.0;
    double time_find_suitable = 0.0;
    double time_pivot = 0.0;
    double time_update_assignment = 0.0;
    double time_solve = 0.0;

   private:
    Solver<T> &solver;
  };
}  // namespace solver

#endif /* SOLVERPROFILER_H_ */
