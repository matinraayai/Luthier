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
	InstPrinter() {}
	InstPrinter(elfio::File *file) : file(file) {}
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
			if (o.literalConstant == 0xffffffff)
			{
				return "-1";
			}
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
				sRegNum.push_back(reg.RegIndex() + o.regCount - 1);
				return stream.str();
			}
			else if (reg.IsVReg())
			{
				stream << "v[" << reg.RegIndex() << ":"
					   << reg.RegIndex() + o.regCount - 1 << "]";
				vRegNum.push_back(reg.RegIndex() + o.regCount - 1);
				return stream.str();
			}
			else if (reg.name.find("lo") != std::string::npos)
			{
				return reg.name.substr(0, reg.name.length() - 3);
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

	std::string sopkString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->dst) << ", "
			   << operandString(i->simm16);
		return stream.str();
	}

	std::string sop1String(Inst *i)
	{
		std::stringstream stream;
		if (i->instType.opcode == 28) //s_getpc
		{
			stream << i->instType.instName << " " << operandString(i->dst);
			return stream.str();
		}
		if (i->instType.opcode == 29) //s_setpc
		{
			stream << i->instType.instName << " " << operandString(i->src0);
			return stream.str();
		}
		stream << i->instType.instName << " " << operandString(i->dst) << ", "
			   << operandString(i->src0);
		return stream.str();
	}

	std::string sopcString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->src0) << ", "
			   << operandString(i->src1);
		return stream.str();
	}

	std::string soppString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName;

		if (i->instType.opcode == 12) //s_waitcnt
		{
			if (i->VMCNT != 63)
				stream << " vmcnt(" << i->VMCNT << ")";
			if (i->EXPCNT != 7)
				stream << " expcnt(" << i->EXPCNT << ")";
			if (i->LKGMCNT != 15)
				stream << " lgkmcnt(" << i->LKGMCNT << ")";
		}
		else if (i->instType.opcode >= 2 && i->instType.opcode <= 9) // Branch
		{
			bool symbolFound = false;

			int16_t imm = int16_t(uint16_t(i->simm16.intValue));
			uint64_t target = i->PC + uint64_t(imm * 4) + 4;

			if (file)
			{
				auto symbols = file->GetSymbols();
				for (int i = 0; i < symbols.size(); i++)
				{
					if (symbols.at(i)->value == target)
					{
						stream << " " << symbols.at(i)->name;
						symbolFound = true;
					}
				}
				if (!symbolFound)
				{
					stream << " couldn't find target";
				}
			}
			else
			{
				printf("File not found\n");
			}
		}
		else if (i->instType.opcode == 1 || i->instType.opcode == 10)
		{
			return stream.str(); //return empty string
		}
		else
		{
			stream << " " << operandString(i->simm16);
		}
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

	// sdwaSelectString stringify SDWA select types
	std::string sdwaSelectString(int sdwaSel)
	{
		std::string s;
		switch (sdwaSel)
		{
		case 0:
			s = "BYTE_0";
			break;
		case 1:
			s = "BYTE_1";
			break;
		case 2:
			s = "BYTE_2";
			break;
		case 3:
			s = "BYTE_3";
			break;
		case 4:
			s = "WORD_0";
			break;
		case 5:
			s = "WORD_1";
			break;
		case 6:
			s = "DWORD";
			break;
		default:
			s = "unknown SDWASelect type";
			break;
		}
		return s;
	}

	std::string sdwaUnusedString(int sel)
	{
		std::string s;
		switch (sel)
		{
		case 0:
			s = "UNUSED_PAD";
			break;
		case 1:
			s = "UNUSED_SEXT";
			break;
		case 2:
			s = "UNUSED_PRESERVE";
			break;
		default:
			s = "unknown SDWAUnused type";
			break;
		}
		return s;
	}
	
	std::string sdwaString(Inst *i)
	{
		std::stringstream stream;

		stream << " dst_sel:" << sdwaSelectString(i->DstSel);

		stream << " dst_unused:" << sdwaUnusedString(i->DstUnused);

		stream << " src0_sel:" << sdwaSelectString(i->Src0Sel);

		if(i->format.formatType == VOP2)
			stream << " src1_sel:" << sdwaSelectString(i->Src1Sel);

		return stream.str();
	}

	std::string vop1String(Inst *i)
	{
		std::stringstream stream;
		std::string suffix;
		i->IsSdwa ? suffix = "_sdwa" : suffix = "_e32";	
		
		stream << i->instType.instName << suffix << " " << operandString(i->dst) << ", "
			   << operandString(i->src0);

		if (i->IsSdwa) stream << sdwaString(i);

		return stream.str();
	}

	std::string vop2String(Inst *i)
	{
		std::stringstream stream;
		std::string suffix;

		i->IsSdwa ? suffix = "_sdwa" : suffix = "_e32";	
		stream << i->instType.instName << suffix << " " << operandString(i->dst);

		if (i->instType.opcode <= 30 && i->instType.opcode >= 15)
		{
			if (i->instType.opcode != 17 && i->instType.opcode != 20)
			{
				stream << ", vcc";
			}
		}
		
		stream << ", " << operandString(i->src0) << ", " << operandString(i->src1);

		if (i->IsSdwa) stream << sdwaString(i);

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

	std::string vop3pString(Inst *i)
	{
		std::stringstream stream;
	
		//Here, I'm hard coding cases where the instruction uses archVgpr
		//This is BADm because there are A LOT if instructions that use these registers
		//So, in the future we need to update reg.h with the archVgpr.
		if (i->instType.opcode == 88) //V_ACCVGPR_READ
		{
			auto src0String = operandString(i->src0);
			src0String.replace(0, 1, "a");
		
			stream << i->instType.instName << "_b32 ";
			stream << operandString(i->dst) << ", ";
			stream << src0String;
		}
		else if (i->instType.opcode == 89) //V_ACCVGPR_WRITE
		{
			auto dstString = operandString(i->dst);
			dstString.replace(0, 1, "a");

			stream << i->instType.instName << "_b32 ";
			stream << dstString << ", ";
			stream << operandString(i->src0);
		}
		else
		{
			stream << i->instType.instName << " ";
			stream << operandString(i->dst) << ", ";
			stream << operandString(i->src0) << ", ";
		}

		if (i->instType.opcode < 64)
		{
			stream << operandString(i->src1);
		}

		if ((i->instType.opcode > 31 && i->instType.opcode < 44) 
      		|| (i->instType.opcode == 0) 
      		|| (i->instType.opcode == 9) 
      		|| (i->instType.opcode == 14))
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
		case SOPK:
			return sopkString(i);
		case SOP1:
			return sop1String(i);
		case SOPC:
			return sopcString(i);
		case SOPP:
			return soppString(i);
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
		case VOP3P:
			return vop3pString(i);
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