#ifndef INSTR_H
#define INSTR_H
#include "luthier_types.hpp"
#include <hsa/hsa.h>
#include <fmt/ostream.h>

#include <string>
#include <vector>

// TODO: Make Instruction an opaque object
// TODO: Make Instr's constructor protected. Make Disassembler a friend of the Instr class
/////* Instruction class returned by the NVBit inspection API nvbit_get_instrs */
////class Instr {
//// public:
////  /* all supported arch have at most 255 general purpose registers */
////  static constexpr const int RZ = 255;
////  /* the always true predicate is indicated as "7" on all the archs */
////  static constexpr const int PT = 7;
////  /* the entire predicate register is ecoded as "8" */
////  static constexpr const int PR = 8;
////  static constexpr const int URZ = 63;
////  static constexpr const int UPT = 7;  // uniform predicate true
////  static constexpr const int UPR = 8;  // entire uniform predicate register
////  static constexpr const int MAX_CHARS = 256;
////
////  enum class memOpType {
////    NONE,
////    LOCAL,     // local memory operation
////    GENERIC,   // generic memory operation
////    GLOBAL,    // global memory operation
////    SHARED,    // shared memory operation
////    CONSTANT,  // constant memory operation
////    GLOBAL_TO_SHARED, // read from global memory then write to shared memory
////  };
////  static constexpr const char* memOpTypeStr[] = {
////      "NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT",
////      "GLOBAL_TO_SHARED"};
////
////  enum class operandType {
////    IMM_UINT64,
////    IMM_DOUBLE,
////    REG,
////    PRED,
////    UREG,
////    UPRED,
////    CBANK,
////    MREF,
////    GENERIC
////  };
////
////  static constexpr const char* operandTypeStr[] = {
////      "IMM_UINT64", "IMM_DOUBLE", "REG",  "PRED",   "UREG",
////      "UPRED",      "CBANK",      "MREF", "GENERIC"};
////
////  enum class regModifierType {
////    /* stride modifiers */
////    X1,
////    X4,
////    X8,
////    X16,
////    /* size modifiers */
////    U32,
////    U64,
////    NO_MOD
////  };
////  static constexpr const char* regModifierTypeStr[] = {
////      "X1", "X4", "X8", "X16", "U32", /* no U */ "64", "NO_MOD"};
////
////  typedef struct { int imm; } mref_t;
////
////  typedef struct {
////    /* operand type */
////    operandType type;
////    /* is negative */
////    bool is_neg;
////    /* is not */
////    bool is_not;
////    /* is absolute */
////    bool is_abs;
////
////    union {
////      struct {
////        uint64_t value;
////      } imm_uint64;
////
////      struct {
////        double value;
////      } imm_double;
////
////      struct {
////        int num;
////        /* register properties .XXX */
////        char prop[MAX_CHARS];
////      } reg;
////
////      struct {
////        int num;
////      } pred;
////
////      struct {
////        int id;
////        bool has_imm_offset;
////        int imm_offset;
////        bool has_reg_offset;
////        int reg_offset;
////      } cbank;
////
////      struct {
////        bool has_ra;
////        int ra_num;
////        regModifierType ra_mod;
////        bool has_ur;
////        int ur_num;
////        bool has_imm;
////        int imm;
////      } mref;
////
////      struct {
////        char array[MAX_CHARS];
////      } generic;
////
////    } u;
////  } operand_t;
////
////  /* returns the "string"  containing the SASS, i.e. IMAD.WIDE R8, R8, R9 */
////  const char* getSass();
////  /* returns offset in bytes of this instruction within the function */
////  uint32_t getOffset();
////  /* returns the id of the instruction within the function */
////  uint32_t getIdx();
////  /* returns true if instruction used predicate */
////  bool hasPred();
////  /* returns predicate number, only valid if hasPred() == true */
////  int getPredNum();
////  /* returns true if predicate is negated (i.e. @!P0), only valid if hasPred()
////   * == true */
////  bool isPredNeg();
////  /* if predicate is uniform predicate (e.g., @UP0), only valid if hasPred()
////   * == true */
////  bool isPredUniform();
////  /* returns full opcode of the instruction (i.e. IMAD.WIDE ) */
////  const char* getOpcode();
////  /* returns short opcode of the instruction (i.e. IMAD.WIDE returns IMAD) */
////  const char* getOpcodeShort();
////
////  /* returns memOpType_t */
////  memOpType getMemOpType();
////  bool isLoad();
////  bool isStore();
////  bool isExtended();
////  int getSize();
////
////  /* get number of operands */
////  int getNumOperands();
////  /* get specific operand */
////  const operand_t* getOperand(int num_operand);
////
////  /* print fully decoded instruction */
////  void printDecoded();
////  /* prints one line instruction with idx, offset, sass */
////  void print(const char* prefix = NULL);
////
//// private:
////  /* Constructor used internally by NVBit */
////  Instr();
////  /* Reserved variable used internally by NVBit */
////  const void* reserved;
////  friend class Nvbit;
////  friend class Function;
////};
////
/////* basic block struct */
////typedef struct { std::vector<Instr*> instrs; } basic_block_t;
////
/////* control flow graph struct */
////typedef struct {
////  /* indicates the control flow graph can't be statically predetermined
////   * because the function from which is belong uses jmx/brx types of branches
////   * which targets depends of registers values that are known only
////   * at runtime */
////  bool is_degenerate;
////  /* vector of basic block */
////  std::vector<basic_block_t*> bbs;
////} CFG_t;
//

namespace luthier {

// TODO: Should we move the operand stuff into an operand.cpp/hpp?
/**
 * Describes a single operand in an instruction
 */
class Operand {
 public:
    enum OperandType {
        InvalidOperandType,
        RegOperand,
        SpecialRegOperand,
        WaitCounter,
        ImmOperand,
        LiteralConstant,
        SpecialOperand
    };
    Operand(std::string op_str, OperandType type, int code, 
            float floatValue, long int intValue, uint32_t literalConstant);
    
    [[nodiscard]] std::string getOperand() const;
    [[nodiscard]] OperandType getOperandType() const;
    [[nodiscard]] int getOperandCode() const;
    [[nodiscard]] float getOperandFloatValue() const;
    [[nodiscard]] long int getOperandIntValue() const;
    [[nodiscard]] uint32_t getOperandLiteralConstant() const;
    friend std::ostream& operator<<(std::ostream& os, const Operand& op);

 private:
    std::string operand;
    OperandType operandType;
    int operandCode;
    float operandFloatVal;
    long int operandIntVal;
    uint32_t operandConst;
};

/**
 * @class Instr is an abstraction over ISA located in memory. If the Instr is located inside an executable,
 * it must be backed by an @struct hsa_executable_symbol_t and a (frozen) @struct hsa_executable_t.
 * It can also be backed by host memory only.
 * The human-readable string form of the instruction should only use the std::string class instead.
 */
class Instr {
 public:
    /**
     *
     * @param instStr
     * @param executable
     * @param symbol
     * @param deviceAccessibleInstrAddress
     * @param hostAccessibleInstrAddress
     * @param instrSize
     */
    Instr(std::string instStr, hsa_agent_t agent, hsa_executable_t executable,
          hsa_executable_symbol_t symbol, luthier_address_t deviceAccessibleInstrAddress,
          luthier_address_t hostAccessibleInstrAddress, size_t instrSize) : executable_(executable),
                                                                            hostAddress_(hostAccessibleInstrAddress),
                                                                            deviceAddress_(deviceAccessibleInstrAddress),
                                                                            instStr_(std::move(instStr)),
                                                                            size_(instrSize),
                                                                            agent_(agent),
                                                                            executableSymbol_(symbol){};

    /**
     *
     * @param instStr
     * @param executable
     * @param symbol
     * @param DeviceAccessibleInstrAddress
     * @param instrSize
     */
    Instr(std::string instStr, hsa_agent_t agent, hsa_executable_t executable,
          hsa_executable_symbol_t symbol, luthier_address_t DeviceAccessibleInstrAddress,
          size_t instrSize);

    //TODO: Add more constructors

    //    /**
    //     *
    //     * @param instStr
    //     * @param symbol
    //     * @param DeviceAccessibleInstrAddress
    //     * @param instrSize
    //     */
    //    Instr(std::string instStr,
    //          hsa_executable_symbol_t symbol,
    //          luthier_address_t DeviceAccessibleInstrAddress,
    //          size_t instrSize) : instStr_(std::move(instStr)),
    //                              executableSymbol_(symbol),
    //                              deviceAddress_(DeviceAccessibleInstrAddress),
    //                              size_(instrSize),
    //                              executable_(hsa_executable_t{0}),
    //                              hostAddress_(0) {
    //
    //
    //                              };

    Instr(std::string instStr,
          luthier_address_t instrHostAddress,
          size_t instrSize) : executable_(hsa_executable_t{0}),
                              hostAddress_(instrHostAddress),
                              deviceAddress_(0),
                              instStr_(std::move(instStr)),
                              size_(instrSize),
                              executableSymbol_({0}){};

    const kernel_descriptor_t *getKernelDescriptor();

    luthier_address_t getHostAddress() const;

    [[nodiscard]] hsa_executable_t getExecutable();

    [[nodiscard]] luthier_address_t getDeviceAddress() const { return deviceAddress_; };

    [[nodiscard]] size_t getSize() const { return size_; };

    [[nodiscard]] std::string getInstr() const { return instStr_; };

    [[nodiscard]] hsa_executable_symbol_t getSymbol() const {return executableSymbol_;}
    [[nodiscard]] hsa_agent_t getAgent() const {return agent_;}

    unsigned int getNumOperands();
    int getNumRegsUsed();
    Operand getOperand(int op_num);

    std::vector<Operand> getAllOperands();
    std::vector<Operand> getImmOperands();
    std::vector<Operand> getRegOperands();

    // Functions that edit operands, reutrn TRUE/FALSE on success/fail 
    // These will need to have corresponding functions in the operand class as well
    // reg code has to match what's in the ISA
    // bool changeRegNum(int reg_op_num, int new_reg_code);
    // bool changeImmVal(int imm_op_num, int new_imm_val);

 private:
    hsa_executable_t executable_{};//
    luthier_address_t hostAddress_{};// Host-accessible address of the instruction
    luthier_address_t deviceAddress_;// Device-accessible address of the instruction
    std::string instStr_;
    size_t size_;
    hsa_agent_t agent_{};
    const hsa_executable_symbol_t executableSymbol_;

    std::vector<Operand> operands;
    void GetOperandsFromString();
    static Operand EncodeOperand(std::string op);
};

}// namespace luthier

template <> struct fmt::formatter<luthier::Operand> : fmt::ostream_formatter {};


#endif
