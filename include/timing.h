#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <cstdio>
#include <sys/time.h>

inline double cpu_second() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#endif /* TIMING_HPP_ */
