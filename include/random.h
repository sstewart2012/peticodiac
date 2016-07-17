#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <vector>

namespace {

float* random_float(const unsigned int count, const int low, const int high) {
  float *v = new float[count];
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(low, high);
  for (int i = 0; i < count; ++i) {
    const int val = distr(eng);
    v[i] = (float) val;
  }
  return v;
}

std::vector<float> random_instance(const unsigned int count, const int low, const int high) {
  std::vector<float> v;
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(low, high);
  for (int i = 0; i < count; ++i) {
    const int val = distr(eng);
    //printf("[%d] %d\n", i + 1, val);
    v.push_back((float) val);
  }
  return v;
}

} // namespace

#endif /* RANDOM_HPP */
