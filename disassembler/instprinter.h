#ifndef INSTPRINTER_H
#define INSTPRINTER_H
#include "elf.h"
#include "inst.h"
#include "operand.h"
#include <iomanip>
#include <sstream>
#include <string>
struct InstPrinter
{
	elfio::File *file;
	std::string operandString(Operand o)
	{
		std::stringstream stream;
		switch (o.operandType)
		{
		case RegOperand:
			return regOperandToString(o);
		case IntOperand:
			return std::to_string(o.intValue);
		case FloatOperand:
			return std::to_string(o.floatValue);
		case LiteralConstant:
			stream << "0x" << std::hex << o.literalConstant;
			return stream.str();
		default:
			return "";
		}
	}

	std::string regOperandToString(Operand o)
	{
		Reg reg = o.reg;
		std::stringstream stream;
		if (o.regCount > 1)
		{
			if (reg.IsSReg())
			{
				stream << "s[" << reg.RegIndex() << ":"
					   << reg.RegIndex() + o.regCount - 1 << "]";
				return stream.str();
			}
			else if (reg.IsVReg())
			{
				stream << "v[" << reg.RegIndex() << ":"
					   << reg.RegIndex() + o.regCount - 1 << "]";
				return stream.str();
			}
			else if (reg.name.find("lo") != std::string::npos)
			{
				return reg.name.substr(0, reg.name.length() - 2);
			}
			return "unknown register";
		}
		return reg.name;
	}

	std::string sop2String(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->dst) << ", "
			   << operandString(i->src0) << ", " << operandString(i->src1);
		return stream.str();
	}

	std::string sop1String(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->dst) << ", "
			   << operandString(i->src0);
		return stream.str();
	}

	std::string smemString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->data) << ", "
			   << operandString(i->base) << ", 0x" << std::hex
			   << uint16_t(i->offset.intValue);
		return stream.str();
	}

	std::string vop1String(Inst *i)
	{
		std::stringstream stream;
		std::string suffix;
		suffix = "_e32";
		stream << i->instType.instName << suffix << " " << operandString(i->dst) << ", "
			   << operandString(i->src0);
		return stream.str();
	}

	std::string vop2String(Inst *i)
	{
		std::stringstream stream;
		std::string suffix;
		suffix = "_e32";
		stream << i->instType.instName << suffix << " " << operandString(i->dst);
		if (i->instType.opcode <= 30 && i->instType.opcode >= 15)
		{
			if (i->instType.opcode != 17)
			{
				stream << ", vcc";
			}
		}
		stream << ", " << operandString(i->src0) << ", " << operandString(i->src1);
		if (i->instType.opcode == 0 || i->instType.opcode == 28 || i->instType.opcode == 29)
		{
			stream << ", vcc";
		}
		else if (i->instType.opcode == 24 || i->instType.opcode == 37)
		{
			stream << ", " << operandString(i->src2);
		}
		return stream.str();
	}

	std::string vopcString(Inst *i)
	{
		std::stringstream stream;
		std::string dst = "vcc";
		if (i->instType.instName.find("cmpx") != std::string::npos)
		{
			dst = "exec";
		}
		stream << i->instType.instName << "_e32 " << dst << ", " << operandString(i->src0) << ", " << operandString(i->src1);
		return stream.str();
	}

	std::string vop3aString(Inst *i)
	{
		std::stringstream stream;
		std::string suffix = "";
		if (i->instType.opcode == 196 || i->instType.opcode == 256)
		{
			suffix += "_e64";
		}
		stream << i->instType.instName + suffix << " " << operandString(i->dst);
		stream << ", " << vop3aInputoperandString(i, i->src0, i->Src0Neg, i->Src0Abs);
		stream << ", " << vop3aInputoperandString(i, i->src1, i->Src1Neg, i->Src1Abs);

		if (i->instType.opcode == 256)
		{
			stream << ", vcc";
		}

		// if (i->instType.opcode == 511)
		// {
		// 	stream << ", -1";
		// }

		if (i->instType.SRC2Width == 0)
		{
			return stream.str();
		}

		stream << ", " << vop3aInputoperandString(i, i->src2, i->Src2Neg, i->Src2Abs);
		return stream.str();
	}

	std::string vop3aInputoperandString(Inst *i, Operand o, bool neg, bool abs)
	{
		std::stringstream s;
		if (neg)
		{
			s << "-";
		}
		if (abs)
		{
			s << "|";
		}
		s << operandString(o);
		if (abs)
		{
			s << "|";
		}
		return s.str();
	}

	std::string vop3bString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " ";

		if (i->instType.opcode > 255)
		{
			stream << operandString(i->dst) << ", ";
		}
		stream << operandString(i->sdst) << ", " << operandString(i->src0) << ", " << operandString(i->src1);

		if (i->instType.opcode > 255 && i->instType.opcode != 281 && i->instType.SRC2Width > 0)
		{
			stream << ", " << operandString(i->src2);
		}
		return stream.str();
	}
	std::string flatString(Inst *i)
	{
		std::stringstream stream;

		switch (i->Seg)
		{
		case 0:
			stream << "flat";
		case 1:
			stream << "scratch";
		case 2:
			stream << "global";
		}
		auto opcode = i->instType.opcode;
		if (opcode >= 16 && opcode <= 23)
		{
			stream << i->instType.instName << " " << operandString(i->dst) << ", " << operandString(i->addr);
		}
		else if (opcode >= 24 && opcode <= 31)
		{
			stream << i->instType.instName << " " << operandString(i->addr) << ", " << operandString(i->data);
		}

		if (i->Seg == 2)
		{
			if (opcode >= 16 && opcode <= 18 || opcode == 20 || opcode == 21 || opcode == 23 || opcode == 28 || opcode == 31)
			{
				stream << ", off";
				if (i->offset.intValue != 0)
				{
					stream << " offset:" << (int)i->offset.intValue;
				}
			}
		}
		return stream.str();
	}

	std::string dsString(Inst *i)
	{
		std::stringstream stream;
		auto oc = i->instType.opcode;
		stream << i->instType.instName << " ";
		if (oc >= 54 && oc <= 60 || oc >= 118 && oc <= 120 || oc >= 254 && oc <= 255)
		{
			stream << operandString(i->dst) << ", ";
		}
		stream << operandString(i->addr);

		if (i->instType.SRC0Width > 0)
		{
			stream << ", " << operandString(i->data);
		}

		if (i->instType.SRC1Width > 0)
		{
			stream << ", " << operandString(i->data1);
		}

		if (oc == 13 || oc == 54 || oc == 254 || oc == 155)
		{
			if (i->Offset0 > 0)
			{
				stream << " offset0:" << i->Offset0;
			}
		}
		else
		{
			if (i->Offset0 > 0)
			{
				stream << " offset0:" << i->Offset0;
			}
			if (i->Offset1 > 0)
			{
				stream << " offset1:" << i->Offset1;
			}
		}
		return stream.str();
	}

	std::string print(Inst *i)
	{
		switch (i->format.formatType)
		{
		case SOP2:
			return sop2String(i);
		case SOP1:
			return sop1String(i);
		case SMEM:
			return smemString(i);
		case VOP1:
			return vop1String(i);
		case VOP2:
			return vop2String(i);
		case VOPC:
			return vopcString(i);
		case VOP3a:
			return vop3aString(i);
		case VOP3b:
			return vop3bString(i);
		case FLAT:
			return flatString(i);
		case DS:
			return dsString(i);
		default:
			return std::string("");
			// throw std::runtime_error("unknown instruction format type");
		}
	}
};
#endif