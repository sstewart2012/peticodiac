#ifndef CPUSOLVER_H_
#define CPUSOLVER_H_

#include <cmath>
#include <cstring>
#include <omp.h>

#include "Solver.h"

namespace solver {
class CpuSolver : public Solver {
  public:
    CpuSolver(const int num_vars, const int max_num_constrs);
    ~CpuSolver();
    virtual bool add_constraint(const std::vector<float> constr) override;
    virtual void set_bounds(const int idx, const float lower, const float upper) override;
    virtual std::vector<float> solution() const override;
    virtual void print_tableau() const override;
    virtual void print_variables() const override;
    virtual inline int num_problem_vars() const override {
      return ncols_;
    }
    virtual inline int num_additional_vars() const override {
      return nrows_;
    }
    virtual inline int num_constraints() const override {
      return next_constr_idx_;
    }
    virtual inline int num_vars() const override {
      return ncols_ + next_constr_idx_;
    }

   protected:
    virtual bool is_broken(const int idx) const;
    virtual bool check_bounds(int &broken_idx) override;
    virtual bool find_suitable(const int broken_idx, int &suitable_idx) override;
    virtual void pivot(const int broken_idx, const int suitable_idx) override;
    virtual float compute_assignment(const int idx) const;
    virtual void swap(const int row, const int col, const int basic_idx, const int nonbasic_idx);
    virtual float get_assignment(const int idx) const;
    virtual inline void update_assignment() override {
      incr_step_count();
    }
    virtual inline float get_lower(const int idx) const {
      return lower_[idx];
    }
    virtual inline float get_upper(const int idx) const {
      return upper_[idx];
    }

  private:
    bool find_suitable_increase(const int broken_idx, int &suitable_idx, const float delta);
    bool find_suitable_decrease(const int broken_idx, int &suitable_idx, const float delta);
    void pivot_update_inner(const float alpha, const int row, const int col);
    void pivot_update_row(const float alpha, const int row);
    void pivot_update_column(const float alpha, const int col);

  protected:
    const int ncols_;
    const int nrows_;
    int next_constr_idx_ = 0;
    int* const col_to_var_ = new int[ncols_];
    int* const row_to_var_ = new int[nrows_];
    int* const var_to_tableau_ = new int[nrows_ + ncols_];
    float* const tableau_ = new float[nrows_ * ncols_];
    float* const assigns_ = new float[nrows_ + ncols_];
    float* const lower_ = new float[nrows_ + ncols_];
    float* const upper_ = new float[nrows_ + ncols_];
    std::set<int> basic_;
    std::set<int> nonbasic_;
    std::map<int, int> map_assigns_;
  };
}  // namespace solver

#endif /* CPUSOLVER_H_ */
