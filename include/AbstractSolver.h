#ifndef ABSTRACTSOLVER_H_
#define ABSTRACTSOLVER_H_

#include <string>
#include <map>
#include <set>
#include <vector>
#include <cassert>
#include "timing.h"

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

  /**
   * AbstractSolver
   */
  class AbstractSolver {
   public:
    virtual ~AbstractSolver() {
    }

    virtual bool add_constraint(const std::vector<float> constr) = 0;
    virtual void set_bounds(const int idx, const float lower,
                            const float upper) = 0;
    virtual std::vector<float> solution() const = 0;

    virtual bool solve() {
      double time = -cpu_second();
      int broken_idx = -1;
      int suitable_idx = -1;
      while (!check_bounds(broken_idx)) {
        if (!find_suitable(broken_idx, suitable_idx))
          return false;
        pivot(broken_idx, suitable_idx);
        update_assignment();
        if (get_step_count() % 1000 == 0)
          printf("%d steps\n", get_step_count());
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
  }
  ;

  /**
   * Solver
   */
  class Solver : public AbstractSolver {
    friend class SolverProfiler;
   public:

    // Factory method
    static Solver* create(SolverType type, const int num_vars,
                          const int max_num_constrs);

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
      const int ncols = num_problem_vars();
      if (idx < ncols)
        return "x" + std::to_string(idx);
      else
        return "s" + std::to_string(idx - ncols);
    }

   private:
    int steps_ = 0;
  };

  class CpuSolver : public Solver {
   public:
    CpuSolver(const int num_vars, const int max_num_constrs);
    ~CpuSolver();
    virtual bool add_constraint(const std::vector<float> constr) override;
    virtual void set_bounds(const int idx, const float lower, const float upper)
        override;
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
    virtual void swap(const int row, const int col, const int basic_idx,
                      const int nonbasic_idx);

    virtual inline void update_assignment() override {
      incr_step_count();
    }

    virtual float get_assignment(const int idx) const {
      assert(idx < num_vars());
      if (basic_.find(idx) == basic_.end())
        return assigns_[idx];
      int tmp = (*map_assigns_.find(idx)).second;
      if (tmp == get_step_count())
        return assigns_[idx];
      else if (basic_.find(idx) == basic_.end())
        return assigns_[idx];
      else
        return compute_assignment(idx);
    }

    virtual inline float get_lower(const int idx) const {
      return lower_[idx];
    }
    virtual inline float get_upper(const int idx) const {
      return upper_[idx];
    }

   private:
    bool find_suitable_increase(const int broken_idx, int &suitable_idx,
                                const float delta);
    bool find_suitable_decrease(const int broken_idx, int &suitable_idx,
                                const float delta);
    void pivot_update_inner(const float alpha, const int row, const int col);
    void pivot_update_row(const float alpha, const int row);
    void pivot_update_column(const float alpha, const int col);

   protected:
    const int nrows_;
    const int ncols_;
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

  class CpuEagerSolver : public CpuSolver {
   public:
    CpuEagerSolver(const int num_vars, const int max_num_constrs)
        : CpuSolver(num_vars, max_num_constrs) {

    }
    virtual ~CpuEagerSolver() {
    }

   protected:
    virtual void update_assignment() override;

    virtual inline float get_assignment(const int idx) const override {
      assert(idx < num_vars());
      return assigns_[idx];
    }

  };

  #ifdef CUDA_ENABLED
  class CudaSolver : public CpuSolver {
   public:
    CudaSolver(const int num_vars, const int num_constrs, const int device_id);
    virtual ~CudaSolver();

    virtual bool solve() {
      pre_solve();
      return CpuSolver::solve();
    }

   protected:
    virtual bool find_suitable(const int broken_idx, int &suitable_idx) override;
    virtual void pivot(const int broken_idx, const int suitable_idx) override;
    virtual void swap(const int row, const int col, const int basic_idx,
                      const int nonbasic_idx) override;
    virtual float compute_assignment(const int idx) const override;

   private:
    virtual void pre_solve();

   private:
    const int device_id_;
    cudaDeviceProp prop_;
    float* d_tableau_ = nullptr;
    float* d_tableau_row_ = nullptr;
    float* d_lower_ = nullptr;
    float* d_upper_ = nullptr;
    float* d_assigns_ = nullptr;
    int* d_col_to_var_ = nullptr;
  };

  class SolverProfiler : public Solver {
   public:
    SolverProfiler(Solver &solver)
        : solver(solver) {

    }
    virtual ~SolverProfiler() {

    }

    virtual bool add_constraint(const std::vector<float> constr) {
      return solver.add_constraint(constr);
    }

    virtual void set_bounds(const int idx, const float lower, const float upper) {
      return solver.set_bounds(idx, lower, upper);
    }

    virtual std::vector<float> solution() const {
      return solver.solution();
    }

    virtual int num_problem_vars() const {
      return solver.num_problem_vars();
    }

    virtual int num_additional_vars() const {
      return solver.num_additional_vars();
    }

    virtual int num_vars() const {
      return solver.num_vars();
    }

    virtual int num_constraints() const {
      return solver.num_constraints();
    }

    virtual void print_tableau() const {
      return solver.print_tableau();
    }

    virtual void print_variables() const {
      return solver.print_variables();
    }

    virtual int get_step_count() const override {
      return solver.get_step_count();
    }

    virtual bool solve() {
      double time = -cpu_second();
      bool result = Solver::solve();
      time += cpu_second();
      time_solve += time;
      return result;
    }

   protected:

    virtual bool check_bounds(int &broken_idx) override {
      double time = -cpu_second();
      bool result = solver.check_bounds(broken_idx);
      time += cpu_second();
      time_check_bounds += time;
      return result;
    }

    virtual bool find_suitable(const int broken_idx, int &suitable_idx) override {
      double time = -cpu_second();
      bool result = solver.find_suitable(broken_idx, suitable_idx);
      time += cpu_second();
      time_find_suitable += time;
      return result;
    }

    virtual void pivot(const int broken_idx, const int suitable_idx) override {
      double time = -cpu_second();
      solver.pivot(broken_idx, suitable_idx);
      time += cpu_second();
      time_pivot += time;
    }

    virtual void update_assignment() {
      double time = -cpu_second();
      solver.update_assignment();
      time += cpu_second();
      time_update_assignment += time;
    }

   public:
    double time_check_bounds = 0.0;
    double time_find_suitable = 0.0;
    double time_pivot = 0.0;
    double time_update_assignment = 0.0;
    double time_solve = 0.0;

   private:
    Solver &solver;
  };

  #endif /* CUDA_ENABLED */

}  // namespace solver

#endif /* ABSTRACTSOLVER_H_ */
