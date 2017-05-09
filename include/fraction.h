#ifndef FRACTION_H_
#define FRACTION_H_

#include <iostream>
using namespace std;

namespace solver {
  class Fraction {
  public:
    Fraction();
    Fraction(long long int numerator);
    Fraction(long long int numerator, long long int denominator);
    ~Fraction();
    friend std::ostream& operator<<(std::ostream& os, const Fraction obj);
    Fraction operator*(const Fraction& operand) const;
    Fraction operator/(const Fraction& operand) const;
    Fraction operator+(const Fraction& operand) const;
    Fraction operator-(const Fraction& operand) const;
    Fraction operator-() const;
    Fraction& operator=(const Fraction& operand);
    Fraction& operator+=(const Fraction& operand);
    Fraction& operator-=(const Fraction& operand);
    bool operator<(const Fraction& operand) const;
    bool operator<=(const Fraction& operand) const;
    bool operator>(const Fraction& operand) const;
    bool operator>=(const Fraction& operand) const;
    bool operator!=(const long long int operand) const;
    bool operator==(const Fraction& operand) const;
    bool operator==(const long long int operand) const;
    void reduce(Fraction& fraction) const;

  private:
    long long int numerator;
    long long int denominator;
    long long int gcd(long long int num1, long long int num2) const;
  };

  Fraction operator/(long long int operand1, const Fraction& operand2);
}

#endif /* FRACTION_H_ */
