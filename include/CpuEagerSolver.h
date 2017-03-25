#ifndef CPUEAGERSOLVER_H_
#define CPUEAGERSOLVER_H_

#include <omp.h>
#include "CpuSolver.h"

namespace solver {
  class CpuEagerSolver : public CpuSolver {
   public:
    CpuEagerSolver(const int num_vars, const int max_num_constrs);
    virtual ~CpuEagerSolver();

   protected:
    virtual void update_assignment() override;
    virtual inline float get_assignment(const int idx) const override {
      assert(idx < num_vars());
      return assigns_[idx];
    }
  };
}  // namespace solver

#endif /* CPUEAGERSOLVER_H_ */
