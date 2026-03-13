//------------------------------------------------------------------------------
// Convenience Functions
//------------------------------------------------------------------------------

class CalculateSingleOffsetDSAddr<dag addr, dag offset,
                                  LLVMQualPointerType localOrRegionPtr> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr), offset);
}

class CalculateDoubleOffsetDSAddr<dag addr, dag offset0, dag offset1,
                                  LLVMQualPointerType localOrRegionPtr,
                                  LLVMType dataType> {
  int Adj = !cond(!eq(dataType.VT.Size, 64) : 8,
                  (!or(!eq(dataType.VT.Size, 4),
                       !eq(dataType.VT.Size, 8),
                       !eq(dataType.VT.Size, 16))
                       ) : 4);
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr),
                         (LLVMAdd (LLVMMul offset0, Adj),
                                   (LLVMMul offset1, Adj)
                                   )
                         );
}

def DS_ADD_U32_sem : InstSISemantic<DS_ADD_U32> {
  let Semantic = [
     (DefVal $tmp, (LLVMLoad llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret)),
     (LLVMStore (LLVMAdd (GetVal $tmp), (GetNamedOperand $data)))
  ];
}

def DS_SUB_U32_sem : InstSISemantic<DS_SUB_U32> {
  let Semantic = [
     (DefVal $tmp, (LLVMLoad llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret)),
     (LLVMStore (LLVMSub (GetVal $tmp), (GetNamedOperand $data)))
  ];
}

def DS_RSUB_U32_sem : InstSISemantic<DS_RSUB_U32> {
  let Semantic = [
     (DefVal $tmp, (LLVMLoad llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret)),
     (LLVMStore (LLVMSub (GetNamedOperand $data), (GetVal $tmp)))
  ];
}

def DS_INC_U32_sem : InstSISemantic<DS_INC_U32> {
  let Semantic = [
     (DefVal $tmp, (LLVMLoad llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret)),
     (LLVMStore (LLVMSelect (LLVMCmp SETGE, (GetVal $tmp), (GetNamedOperand $data)),
                            (ConstantInt llvm_i32_ty, 0), (LLVMAdd (GetVal $tmp), (ConstantInt llvm_i32_ty, 1))))
  ];
}