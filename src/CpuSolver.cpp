#include "CpuSolver.h"
#include <typeinfo>

namespace solver {
template <typename T>
CpuSolver<T>::CpuSolver(const int num_vars, const int max_num_constrs)
    : ncols_(num_vars), nrows_(max_num_constrs) {
  memset(tableau_, 0, nrows_ * ncols_ * sizeof(T));
  memset(verify_tableau_, 0, nrows_ * ncols_ * sizeof(T));
  int i, j;
  for (i = 0; i < ncols_; ++i) {
    col_to_var_[i] = i;
    var_to_tableau_[i] = i;
    lower_[i] = 0.0f;
    upper_[i] = NO_BOUND;
    assigns_[i] = 0.0f;
    nonbasic_.insert(i);
    map_assigns_[i] = 0;
    verify_lower_[i] = 0.0f;
    verify_upper_[i] = NO_BOUND;
  }
  for (j = 0; j < nrows_; ++j, ++i) {
    row_to_var_[j] = i;
    var_to_tableau_[i] = j;
    lower_[i] = 0.0f;
    upper_[i] = 0.0f;
    assigns_[i] = 0.0f;
    basic_.insert(i);
    map_assigns_[i] = 0;
    verify_lower_[i] = 0.0f;
    verify_upper_[i] = 0.0f;
    verify_basic_.insert(i);
  }
}

template <typename T>
CpuSolver<T>::~CpuSolver() {
  delete[] tableau_;
  delete[] lower_;
  delete[] upper_;
  delete[] assigns_;
  delete[] row_to_var_;
  delete[] col_to_var_;
  delete[] var_to_tableau_;
  delete[] verify_tableau_;
  delete[] verify_lower_;
  delete[] verify_upper_;
}

template <typename T>
bool CpuSolver<T>::add_constraint(const std::vector<T> constr) {
#ifdef DEBUG
  std::cout << "add_constraint() constr size = " << constr.size() << " and ncols = " << ncols_ << std::endl;
  //printf("add_constraint() constr size = %u and ncols = %u\n", constr.size(), ncols_);
#endif
  if (int(constr.size()) != ncols_) {
    return false;
  }
  std::memcpy(&tableau_[next_constr_idx_ * ncols_], &constr[0], constr.size() * sizeof(T));
  std::memcpy(&verify_tableau_[next_constr_idx_ * ncols_], &constr[0], constr.size() * sizeof(T));
  next_constr_idx_++;
  return true;
}

template <typename T>
void CpuSolver<T>::set_bounds(const int idx, const T lower, const T upper) {
  lower_[idx] = lower;
  upper_[idx] = upper;
  verify_lower_[idx] = lower;
  verify_upper_[idx] = upper;
}

template <typename T>
std::vector<T> CpuSolver<T>::solution() const {
  std::vector<T> s;
  for (int i = 0; i < ncols_; ++i)
    s.push_back(assigns_[i]);
  return s;
}

template <typename T>
void CpuSolver<T>::print_tableau() const {
  for (auto& i: basic_) {
    std::cout << "b" << var_to_tableau_[i] << " " << i << ":[" << this->var2str(i).c_str() << "] ";
    std::cout << lower_[i] << " <= " << assigns_[i] << " <= " << upper_[i];
    if (is_broken(i)) {
      std::cout << " broken";
    }
    std::cout << std::endl;
    // printf("b%d %d:[%s] %.3f <= %.3f <= %.3f%s\n", var_to_tableau_[i], i,
    //        this->var2str(i).c_str(), lower_[i], assigns_[i], upper_[i],
    //        is_broken(i) ? " broken" : "");
  }
  for (auto& i: nonbasic_) {
    std::cout << "n" << var_to_tableau_[i] << " " << i << ":[" << this->var2str(i).c_str() << "] ";
    std::cout << lower_[i] << " <= " << assigns_[i] << " <= " << upper_[i];
    if (is_broken(i)) {
      std::cout << " broken";
    }
    std::cout << std::endl;
    // printf("n%d %d:[%s] %.3f <= %.3f <= %.3f%s\n", var_to_tableau_[i], i,
    //        this->var2str(i).c_str(), lower_[i], assigns_[i], upper_[i],
    //        is_broken(i) ? " broken" : "");
  }
}

template <typename T>
void CpuSolver<T>::print_variables() const {
  for (int i = 0; i < nrows_; ++i) {
    for (int j = 0; j < ncols_; ++j) {
      std::cout << tableau_[i * ncols_ + j];
      if (j < ncols_) {
        std::cout << " ";
      }
      //printf("%.3f%s", tableau_[i * ncols_ + j], j < ncols_ ? " " : "");
    }
    std::cout << endl;
    //printf("\n");
  }
}

template <typename T>
bool CpuSolver<T>::verify_solution() const {
  std::cout << "#### Verify Solution ####" << std::endl;
  int row_index = 0;
  for (auto& i: verify_basic_) {
#ifdef DEBUG
    std::cout << "Expression " << row_index << ": ";
    std::cout << "Lower bound = " << verify_lower_[i] << " | Upper bound = " << verify_upper_[i] << std::endl;
    std::cout << "    ";
    for (int j = 0; j < ncols_; ++j) {
      std::cout << verify_tableau_[row_index * ncols_ + j] << " * ";
      std::cout << "(" << assigns_[j] << ")";
      if (j < ncols_ - 1) {
        std::cout << " + ";
      }
    }
    std::cout << std::endl;
#endif

    T verify_result{};
    for (int j = 0; j < ncols_; ++j) {
      verify_result += verify_tableau_[row_index * ncols_ + j] * assigns_[j];
    }

#ifdef DEBUG
      std::cout << "    Verify result = " << verify_result << std::endl;
#endif

    if (verify_lower_[i] >= 0 && verify_upper_[i] >= 0) {
      if (verify_result < verify_lower_[i] || verify_result > verify_upper_[i]) {
#ifdef DEBUG
        std::cout << "    ERROR: outside the bound!" << std::endl;
#endif
        return false;
      }
    }

    row_index++;
  }

  return true;
}

template <typename T>
bool CpuSolver<T>::is_broken(const int idx) const {
  const T ass = get_assignment(idx);
  const T low = get_lower(idx);
  const T upp = get_upper(idx);
  if (typeid(ass).name() == typeid(float).name()) {
    if ((ass - low) < EPSILON && (ass - low) > -EPSILON) { // "close enough" to lower bound
      return false;
    } else if ((ass - upp) < EPSILON && (ass - upp) > -EPSILON) { // "close enough" to upper bound
      return false;
    } else if (low != NO_BOUND && ass < low) {
      return true;
    } else if (upp != NO_BOUND && ass > upp) {
      return true;
    } else {
      return false;
    }
  } else {
    if (ass == low) {
      return false;
    } else if (ass == upp) {
      return false;
    } else if (low != NO_BOUND && ass < low) {
      return true;
    } else if (upp != NO_BOUND && ass > upp) {
      return true;
    } else {
      return false;
    }
  }
}

template <typename T>
T CpuSolver<T>::get_assignment(const int idx) const {
  assert(idx < num_vars());
  if (basic_.find(idx) == basic_.end())
    return assigns_[idx];
  int tmp = (*map_assigns_.find(idx)).second;
  if (tmp == this->get_step_count())
    return assigns_[idx];
  else if (basic_.find(idx) == basic_.end())
    return assigns_[idx];
  else
    return compute_assignment(idx);
}

template <typename T>
T CpuSolver<T>::compute_assignment(const int idx) const {
  const int rowIdx = var_to_tableau_[idx];
  const T* const row = &tableau_[rowIdx * ncols_];
  T val = 0.0f;
  int i;

  // TODO: Need to reimplement this to replace reduction() as OpenMP reduction is not overloaded in the Fraction class
  //#pragma omp parallel for reduction(+:val)
  #pragma omp parallel for
  for (i = 0; i < ncols_; ++i) {
    val += row[i] * assigns_[col_to_var_[i]];
  }

  // Update the assignment
  const_cast<CpuSolver*>(this)->assigns_[idx] = val;
  const_cast<CpuSolver*>(this)->map_assigns_[idx] = this->get_step_count();

  return val;
}

template <typename T>
bool CpuSolver<T>::check_bounds(int &broken_idx) {
  for (const int& var:basic_) {
    if (is_broken(var)) {
      broken_idx = var;
      return false;
    }
  }
  return true;
}

template <typename T>
bool CpuSolver<T>::find_suitable(const int broken_idx, int &suitable_idx) {
  const T ass = assigns_[broken_idx];
  const T low = lower_[broken_idx];
  const bool increase = ass < low;
  const T delta = increase ? low - ass : ass - upper_[broken_idx];
  if (increase) {
    return find_suitable_increase(broken_idx, suitable_idx, delta);
  } else {
    return find_suitable_decrease(broken_idx, suitable_idx, delta);
  }
}

template <typename T>
bool CpuSolver<T>::find_suitable_increase(const int broken_idx, int &suitable_idx, const T delta) {
  for (const int& var: nonbasic_) {
    const T coeff = tableau_[var_to_tableau_[broken_idx] * ncols_ + var_to_tableau_[var]];
    const T low = lower_[var];
    const T upp = upper_[var];
    const T ass = assigns_[var];
    if ((IS_INCREASABLE(low, upp, ass) && coeff > 0) ||
        (IS_DECREASABLE(low, upp, ass) && coeff < 0)) {
      const T theta = delta / coeff;
      assigns_[var] += coeff < 0 ? -theta : theta;
      assigns_[broken_idx] += delta;
      suitable_idx = var;
      return true;
    }
  }
  return false;
}

template <typename T>
bool CpuSolver<T>::find_suitable_decrease(const int broken_idx, int &suitable_idx, const T delta) {
  for (const int& var: nonbasic_) {
    const T coeff = tableau_[var_to_tableau_[broken_idx] * ncols_ + var_to_tableau_[var]];
    const T low = lower_[var];
    const T upp = upper_[var];
    const T ass = assigns_[var];
    if ((IS_INCREASABLE(low, upp, ass) && coeff < 0) ||
        (IS_DECREASABLE(low, upp, ass) && coeff > 0)) {
      const T theta = delta / coeff;
      assigns_[var] -= coeff < 0 ? theta : -theta;
      assigns_[broken_idx] -= delta;
      suitable_idx = var;
      return true;
    }
  }
  return false;
}

template <typename T>
void CpuSolver<T>::pivot(const int broken_idx, const int suitable_idx) {
  const int pivot_row = var_to_tableau_[broken_idx];
  const int pivot_col = var_to_tableau_[suitable_idx];
#ifdef DEBUG
  std::cout << "Pivot(" << pivot_row << "," << pivot_col << ")" << std::endl;
  //printf("Pivot(%d,%d)\n", pivot_row, pivot_col);
#endif

  // Save the current pivot element (alpha)
  const int alpha_idx = OFFSET(pivot_row, pivot_col, ncols_);
  const T alpha = tableau_[alpha_idx];

  // Update the tableau
  pivot_update_inner(alpha, pivot_row, pivot_col);
  pivot_update_row(alpha, pivot_row);
  pivot_update_column(alpha, pivot_col);
  tableau_[alpha_idx] = 1 / alpha;

  // Swap the basic and nonbasic variables
  swap(pivot_row, pivot_col, broken_idx, suitable_idx);
}

template <typename T>
void CpuSolver<T>::swap(const int row, const int col, const int basic_idx, const int nonbasic_idx) {
  col_to_var_[col] = basic_idx;
  row_to_var_[row] = nonbasic_idx;
  var_to_tableau_[basic_idx] = col;
  var_to_tableau_[nonbasic_idx] = row;
  basic_.erase(basic_idx);
  basic_.insert(nonbasic_idx);
  nonbasic_.erase(nonbasic_idx);
  nonbasic_.insert(basic_idx);
}

template <typename T>
void CpuSolver<T>::pivot_update_inner(const T alpha, const int row, const int col) {
  int i, j;
  #pragma omp parallel for private(i, j)
  for (i = 0; i < nrows_; ++i) {
    if (i == row) {continue;}

    for (j = 0; j < ncols_; ++j) {
      if (j == col) {continue;}
      const int delta_row_idx = i * ncols_;
      const int delta_idx = delta_row_idx + j;
      const T delta = tableau_[delta_idx];
      const T beta = tableau_[row * ncols_ + j];
      const T gamma = tableau_[delta_row_idx + col];
      tableau_[delta_idx] = delta - (beta * gamma) / alpha;
    }
  }
}

template <typename T>
void CpuSolver<T>::pivot_update_row(const T alpha, const int row) {
  T* beta = &tableau_[row * ncols_];
  int i;

  #pragma omp parallel for
  for (i = 0; i < ncols_; ++i) {
    *(beta + i) = -(*(beta + i)) / alpha;
  }
}

template <typename T>
void CpuSolver<T>::pivot_update_column(const T alpha, const int col) {
  T* gamma = &tableau_[col];
  int i;

  #pragma omp parallel for
  for (i = 0; i < nrows_; ++i) {
    *(gamma + (i * ncols_)) = *(gamma + (i * ncols_)) / alpha;
  }
}

template class CpuSolver<float>;
template class CpuSolver<Fraction>;

}  // namespace solver
