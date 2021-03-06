#include <iostream>

// factorial.hs:
//     fac 0 = 1
//     fac n = n * fac (n-1)
//     main = print (fac 6)
// ghc -o factorial factorial.hs
template<int n> struct
fact {
    static const int value = n * fact<n - 1>::value;
};

template<> struct
fact<0> { // specialization for n = 0
    static const int value = 1;
};

// predicate.hs:
//     is_zero 0 = True
//     is_zero x = False
// ghc -o predicate predicate.hs
template<typename T> struct
isPtr {
  static const int value = false;
};

template<typename U> struct
isPtr<U*> {
  static const int value = true;
};

template<typename U> struct
isPtr<U* const> {
    static const bool value = true;
};

template<typename U> struct
isPtr<const U*> {
    static const bool value = false;
};

// count_list.hs
//     l = [1,2,3]
//     count [] = 0
//     count (head:tail) = 1 + count tail
//     main = print (count l)
// ghc count_list.hs
// Just a declaration
template<typename... list> struct
count;

template<> struct
count<> {
  static const int value = 0;
};

template<typename head, typename... tail> struct
count<head, tail...> {
  // "typename... tail" defines a template parameter pack.
  // The expansion is done by following the name of the pack with three dots,
  // as in tail….
  static const int value = 1 + count<tail...>::value;
};

// or_combinator.hs
//     or_combinator f1 f2 = λ x -> (f1 x) || (f2 x)
//     (or_combinator is_zero is_one) 2
// ghc or_combinator.hs
template<template<typename> class f1, template<typename> class f2> struct
or_combinator {
    template<typename T> struct
    lambda {
        static const bool value = f1<T>::value || f2<T>::value;
    };
};

template<typename T> struct
isConst {
  static const int value = false;
};

template<typename U> struct
isConst<const U> {
  static const int value = true;
};

// all.hs
//     all pred [] = True
//     all pred (head:tail) = (pred head) && (all pred tail)
// ghs all.hs
template<template<typename> class predicate, typename... list> struct
all;

template<template<typename> class predicate> struct
all<predicate> {
  static const int value = true;
};

template<template<typename> class predicate, typename head, typename... tail> struct
all<predicate, head, tail...> {
  static const int value = predicate<head>::value && all<predicate, tail...>::value;
};


int main() {
  std::cout << "[factorial.hs] Factorial of 6 = " << fact<6>::value << "\n";
  std::cout << "[predicate.hs] isPtr<int> = " << isPtr<int>::value
            << "; isPtr<char *> = " << isPtr<char *>::value
            << "; isPtr<float * const> = " << isPtr<float * const>::value
            << "; isPtr<float * const> = " << isPtr<const float *>::value
            << "\n";
  std::cout << "[count_list.hs] len of the list of types [int, char, long] = "
            << count<int, char, long>::value << "\n";
  std::cout << "[or_combinator.hs] or_combinator<isPtr, isConst>::lambda<const int>::value = "
            << or_combinator<isPtr, isConst>::lambda<int>::value << ","
            << or_combinator<isPtr, isConst>::lambda<const int>::value << ","
            << or_combinator<isPtr, isConst>::lambda<int *>::value << ","
            << std::endl;
  std::cout << "[all.hs] all<isPtr, int*, char*, float*>::value = "
	    << all<isPtr, int*, char*, float*>::value << "; "
	    << all<isConst, const int*, const char*, float*>::value << std::endl;
  return 0;
}
