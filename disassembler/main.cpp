#include <iostream>
#include "initialize.h"
#include "operand.h"
int main()
{
	initRegs();
	Operand operand;
	newVRegOperand(&operand, 0, 0, 1);
	std::cout << operand.reg->name << "\n";
	// std::cout << VReg(0)->name << "\n";
	return 0;
}
