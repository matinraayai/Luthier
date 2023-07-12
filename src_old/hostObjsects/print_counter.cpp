#include "hip/hip_runtime.h"
#include <iostream>
__managed__ unsigned long counter;
using namespace std;
void sibir_at_term() {
  cout << "Total thread instructions " << counter << "\n";
}
