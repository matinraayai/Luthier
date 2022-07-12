#include <iostream>
#include "initialize.h"
int main()
{
	initRegs();
	std::cout << VReg(0)->name << "\n";
	return 0;
}
