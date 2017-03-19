#ifndef FRACTION_H_
#define FRACTION_H_

#include <iostream>
using namespace std;

namespace solver {
  class Fraction {
  public:
    Fraction();
    Fraction(int numerator);
    Fraction(int numerator, int denominator);
    ~Fraction();
    friend std::ostream& operator<<(std::ostream& os, const Fraction obj);
    Fraction operator*(const Fraction& operand) const;
    Fraction operator/(const Fraction& operand) const;
    Fraction operator+(const Fraction& operand) const;
    Fraction operator-(const Fraction& operand) const;
    Fraction operator-() const;
    Fraction& operator+=(const Fraction& operand);
    Fraction& operator-=(const Fraction& operand);
    bool operator<(const Fraction& operand) const;
    bool operator>(const Fraction& operand) const;
    bool operator!=(const int operand) const;
    bool operator==(const Fraction& operand) const;
    bool operator==(const int operand) const;
    void reduce(Fraction& fraction) const;

  private:
    int numerator;
    int denominator;
    int gcd(int num1, int num2) const;
  };

  Fraction operator/(int operand1, const Fraction& operand2);
}

#endif /* FRACTION_H_ */
