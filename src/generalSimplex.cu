#include <cstdio>
#include <cassert>
#include "generalSimplex.h"

namespace device {

/** Number of work-items is equal to number of rows of the tableau. */
__global__ void check_bounds(const int nrows, const float* const lower,
		const float* const upper, const float* const assigns, int* row_to_var,
		int* const result) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nrows)
		return;

	const int var = row_to_var[tid];
	const float ass = assigns[var];
	const float low = lower[var];
	const float upp = upper[var];

	const bool testA = fabsf(ass - low) < EPSILON;
	const bool testB = fabsf(ass - upp) < EPSILON;
	const bool testC = low != NO_BOUND && ass < low;
	const bool testD = upp != NO_BOUND && ass > upp;

	if (testA || testB || !(testC || testD)) {
		return;
	} else {
		atomicMin(result, var);
	}
}

__global__ void find_suitable(const int ncols, const int broken_idx,
		const float* const tableau, const float* const lower,
		const float* const upper, const float* const assigns,
		const int* const varToTableau, const int* const col_to_var,
		int* const suitable_idx) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ncols)
		return;

	const bool increase = assigns[broken_idx] < lower[broken_idx];
	const int var = col_to_var[tid];
	const float ass = assigns[var];
	const float low = lower[var];
	const float upp = upper[var];
	const float coeff = tableau[varToTableau[broken_idx] * ncols
			+ varToTableau[var]];

	if (increase) {
		if ((IS_INCREASABLE(low, upp, ass) && coeff > 0)
				|| (IS_DECREASABLE(low, upp, ass) && coeff < 0)) {
			atomicMin(suitable_idx, var);
		}
	} else {
		if ((IS_INCREASABLE(low,
				upp, ass) && coeff < 0)
				|| (IS_DECREASABLE(low, upp, ass) && coeff > 0)) {
			atomicMin(suitable_idx, var);
		}
	}
}

__global__ void find_suitable_complete(const int ncols, const int broken_idx,
		const int suitable_idx, const float* const tableau,
		const float* const lower, const float* const upper,
		float* const assigns, const int* const var_to_tableau) {

	if (blockIdx.x * blockDim.x + threadIdx.x > 0)
		return;

	float ass = assigns[broken_idx];
	float low = lower[broken_idx];
	float upp = upper[broken_idx];
	const bool increase = ass < low;
	const float coeff = tableau[OFFSET(var_to_tableau[broken_idx],
			var_to_tableau[suitable_idx], ncols)];

	// Amounts to adjust assignments of suitable and broken variables
	const float delta = increase ? low - ass : ass - upp;
	const float theta = delta / coeff;

	// Read bounds info for the suitable variable to check if
	// increaseable or decreaseable
	ass = assigns[suitable_idx];
	low = lower[suitable_idx];
	upp = upper[suitable_idx];

	if (increase) {
		if ((IS_INCREASABLE(low, upp, ass) && coeff > 0)
				|| (IS_DECREASABLE(low, upp, ass) && coeff < 0)) {
			assigns[suitable_idx] += coeff < 0 ? -theta : theta;
			assigns[broken_idx] += delta;
		}
	} else {
		if ((IS_INCREASABLE(low, upp, ass) && coeff < 0)
				|| (IS_DECREASABLE(low, upp, ass) && coeff > 0)) {
			assigns[suitable_idx] -= coeff < 0 ? theta : -theta;
			assigns[broken_idx] -= delta;
		}
	}
}

__global__ void pivot_update_inner(const float alpha, const int pivot_row,
		const int pivot_col, const int nrows, const int ncols,
		float* const tableau) {
	// Determine thread ID in 2D (x and y)
	const unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; // column index
	const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; // row index

	if (col < ncols && row < nrows && row != pivot_row && col != pivot_col) {
		// Compute helpful indices
		const unsigned int delta_row_idx = OFFSET(row, 0, ncols);
		const unsigned int delta_idx = delta_row_idx + col;

		// Load values from global memory
		const float delta = tableau[delta_idx];
		const float beta = tableau[OFFSET(pivot_row, col, ncols)];
		const float gamma = tableau[delta_row_idx + pivot_col];

		// Store result
		float coeff = delta - (beta * gamma) / alpha;
		tableau[delta_idx] = coeff;
	}
}

__global__ void pivot_update_row(const float alpha, const int row,
		const int ncols, float* const tableau) {
	float* const tableau_row = &tableau[row * ncols];
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= ncols)
		return;
	const float beta = tableau_row[col];
	const float coeff = -beta / alpha;
	tableau_row[col] = coeff;
}

__global__ void pivot_update_column(const float alpha, const int col,
		const int nrows, const int ncols, float* const tableau) {
	float* const tableau_col = tableau + col;
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= nrows)
		return;
	const int idx = row * ncols;
	const float gamma = tableau_col[idx];
	tableau_col[idx] = gamma / alpha;
}

__global__ void update_assignment_row_multiply(const int ncols,
		const int* const col_to_var, const float* const assigns,
		float* const row) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ncols)
		return;
	float* p = row + tid;
	*p *= assigns[col_to_var[tid]];
}

__global__ void update_assignment_1(const int ncols, const int nrows,
		const float* const tableau, const float* const assigns,
		const int* const colToVar, float* const output) {
	extern __shared__ float partial_sums[];
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

	// Boundary check
	if (gid >= ncols)
		return;

	for (int row = 0; row < nrows; ++row) {
		// Pre-fetch and multiply by corresponding assignment
		const float a = assigns[colToVar[gid]];
		partial_sums[lid] = a * tableau[OFFSET(row, gid, ncols)];
		__syncthreads();

		// Reduce using interleaved pairs
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (lid < stride && lid + stride < ncols) {
				partial_sums[lid] += partial_sums[lid + stride];
			}
			__syncthreads();
		}

		// Write the result for this block to global memory
		if (lid == 0) {
			output[OFFSET(row, blockIdx.x, gridDim.x)] = partial_sums[0];
		}
		__syncthreads();
	}
}

__global__ void update_assignment_2(const int n, float* const data) {
	extern __shared__ float partial_sums[];
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

// Boundary check
	if (gid >= n)
		return;

// Pre-fetch
	partial_sums[lid] = data[gid];
	__syncthreads();

// Reduce using interleaved pairs
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (lid < stride && lid + stride < n) {
			partial_sums[lid] += partial_sums[lid + stride];
		}
		__syncthreads();
	}

// Write the result for this block to global memory
	if (lid == 0) {
		data[blockIdx.x] = partial_sums[0];
	}
}

__global__ void update_assignment_complete(const int n, float* const data) {
	extern __shared__ float partial_sums[];
	const int lid = threadIdx.x;
//printf("[%d] var=%d input=%f\n", lid, var, input[idx], input);

// Pre-fetch
	partial_sums[lid] = data[lid];
	__syncthreads();
//printf("[%d] offset=%d var=%d partial_sums=%f\n", lid, offset, var, partial_sums[idx]);

// Reduce
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (lid < stride && lid + stride < n) {
			partial_sums[lid] += partial_sums[lid + stride];
		}
		__syncthreads();
	}

// Write the result to the assignments array
	if (lid == 0) {
		data[0] = partial_sums[0];
	}
}

} // device
