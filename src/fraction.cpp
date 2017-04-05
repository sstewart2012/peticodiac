#include "fraction.h"

namespace solver {
Fraction::Fraction() : numerator(0), denominator(1) {}
Fraction::Fraction(long long int numerator) : numerator(numerator), denominator(1) {}
Fraction::Fraction(long long int numerator, long long int denominator) : numerator(numerator), denominator(denominator) {}
Fraction::~Fraction() {}

std::ostream& operator<<(std::ostream& os, const Fraction obj) {
  os << obj.numerator << "/" << obj.denominator;
  return os;
}

Fraction Fraction::operator*(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.numerator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator/(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator;
  result.denominator = this->denominator * operand.numerator;
  Fraction::reduce(result);
  return result;
}

Fraction operator/(long long int operand1, const Fraction& operand2) {
  Fraction fractionOperand1(operand1);
  Fraction result = fractionOperand1/operand2;
  return result;
}

Fraction Fraction::operator+(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator + operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator-(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator - operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator-() const {
  Fraction result;
  result.numerator = -this->numerator;
  result.denominator = this->denominator;
  Fraction::reduce(result);
  return result;
}

Fraction& Fraction::operator+=(const Fraction& operand) {
  this->numerator = this->numerator * operand.denominator + operand.numerator * this->denominator;
  this->denominator = this->denominator * operand.denominator;
  Fraction::reduce(*this);
  return *this;
}

Fraction& Fraction::operator-=(const Fraction& operand) {
  this->numerator = this->numerator * operand.denominator - operand.numerator * this->denominator;
  this->denominator = this->denominator * operand.denominator;
  Fraction::reduce(*this);
  return *this;
}

bool Fraction::operator<(const Fraction& operand) const {
  long long int leftNumerator = this->numerator * operand.denominator;
  long long int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator < rightNumerator;
}

bool Fraction::operator<=(const Fraction& operand) const {
  long long int leftNumerator = this->numerator * operand.denominator;
  long long int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator <= rightNumerator;
}

bool Fraction::operator>(const Fraction& operand) const {
  long long int leftNumerator = this->numerator * operand.denominator;
  long long int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator > rightNumerator;
}

bool Fraction::operator>=(const Fraction& operand) const {
  long long int leftNumerator = this->numerator * operand.denominator;
  long long int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator >= rightNumerator;
}

bool Fraction::operator!=(const long long int operand) const {
  return this->numerator != this->denominator * operand;
}

bool Fraction::operator==(const Fraction& operand) const {
  return this->numerator * operand.denominator == this->denominator * operand.numerator;
}

bool Fraction::operator==(const long long int operand) const {
  return this->numerator == this->denominator * operand;
}

void Fraction::reduce(Fraction& fraction) const {
  // When the fraction is 0, always reduce the fraction back to 0/1
  if (fraction.numerator == 0) {
    fraction.denominator = 1;
  }
  long long int gcd = Fraction::gcd(fraction.numerator, fraction.denominator);
  fraction.numerator = fraction.numerator / gcd;
  fraction.denominator = fraction.denominator / gcd;
}

long long int Fraction::gcd(long long int num1, long long int num2) const {
  num1 = abs(num1);
  num2 = abs(num2);
  if (num1 < num2) {
    long long int tmp = num1;
    num1 = num2;
    num2 = tmp;
  }

  if (num2 == 0) {
    return num1;
  } else {
    return Fraction::gcd(num2, num1 % num2);
  }
}

}
