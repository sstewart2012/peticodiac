#ifndef generalSimplex_h
#define generalSimplex_h

#define OFFSET(row, col, ncols) (row * ncols + col)
#define NO_BOUND -1
#define EPSILON 0.000001
#define NONBASIC_FLAG 0
#define BASIC_FLAG 1
#define NONE_FOUND -1
#define IS_INCREASABLE(low, upp, ass) (upp == NO_BOUND || ass < upp)
#define IS_DECREASABLE(low, upp, ass) (low == NO_BOUND || ass > low)

namespace device {

__global__ void check_bounds(const int nrows, const float* const lower,
		const float* const upper, const float* const assigns, int* row_to_var,
		int* const result);

__global__ void find_suitable(const int ncols, const int broken_idx,
		const float* const tableau, const float* const lower,
		const float* const upper, const float* const assigns,
		const int* const varToTableau, const int* const colToVar,
		int* const suitable_idx);

__global__ void find_suitable_complete(const int ncols, const int broken_idx,
		const int suitable_idx, const float* const tableau,
		const float* const lower, const float* const upper,
		float* const assigns, const int* const varToTableau);

__global__ void pivot_update_inner(const float alpha, const int pivot_row,
		const int pivot_col, const int nrows, const int ncols,
		float* const tableau);

__global__ void pivot_update_row(const float alpha, const int row,
		const int ncols, float* const tableau);

__global__ void pivot_update_column(const float alpha, const int col,
		const int nrows, const int ncols, float* const tableau);

__global__ void update_assignment_row_multiply(const int ncols,
		const int* const col_to_var, const float* const assigns,
		float* const row);

__global__ void update_assignment_1(const int ncols, const int nrows,
		const float* const tableau, const float* const assigns,
		const int* const colToVar, float* const output);

__global__ void update_assignment_2(const int n, float* const data);

__global__ void update_assignment_complete(const int n, float* const data);

}
#endif
