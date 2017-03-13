#ifndef CPUEAGERSOLVER_H_
#define CPUEAGERSOLVER_H_

#include <omp.h>
#include "CpuSolver.h"

namespace solver {
  template <typename T>
  class CpuEagerSolver : public CpuSolver<T> {
   public:
    CpuEagerSolver(const int num_vars, const int max_num_constrs);
    virtual ~CpuEagerSolver();

   protected:
    virtual void update_assignment() override;
    virtual inline T get_assignment(const int idx) const override {
      assert(idx < this->num_vars());
      return this->assigns_[idx];
    }
  };
}  // namespace solver

#endif /* CPUEAGERSOLVER_H_ */
