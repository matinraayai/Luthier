#include <iostream>
#include "initialize.h"
#include "operand.h"
int main()
{
	initRegs();

	std::cout << newVRegOperand(0, 0, 1).reg.name << "\n";
	// std::cout << VReg(0)->name << "\n";
	return 0;
}
