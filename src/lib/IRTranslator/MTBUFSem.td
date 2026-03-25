//===- MTBUFSem.td ---------------------------------------------*- tblgen -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

//------------------------------------------------------------------------------
// Convenience Functions
//------------------------------------------------------------------------------

class CalculateTypedBufferAddr<dag vaddr, dag rsrc, dag soffset, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), buffer_ptr_type),
                         (LLVMAdd (GetNamedOperand $soffset), (GetNamedOperand $offset))));
}

//------------------------------------------------------------------------------
// MTBUF Instructions — Typed Buffer Load/Store (TBUFFER)
//
// These instructions perform typed buffer (TBUFFER) memory operations with a
// hardware data format conversion specified by the $format field. Because the
// format conversion is performed in hardware and cannot be expressed in the
// current semantic DSL, all entries use empty semantics [].
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Section 1: TBUFFER_LOAD_FORMAT — Typed buffer loads (non-D16)
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 276-339.
//   vega-shader-instruction-set-architecture.pdf, pages 211-283.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 213-285.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 201-271.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 196-264.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 477-556.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 523-604.
//   rdna-shader-instruction-set-architecture.pdf, pages 230-287.
//   rdna2-shader-instruction-set-architecture.pdf, pages 224-281.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 91-564.
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_LOAD_FORMAT_X_OFFSET,
                TBUFFER_LOAD_FORMAT_X_OFFEN,
                TBUFFER_LOAD_FORMAT_X_IDXEN,
                TBUFFER_LOAD_FORMAT_X_BOTHEN,
                TBUFFER_LOAD_FORMAT_X_ADDR64,
                TBUFFER_LOAD_FORMAT_XY_OFFSET,
                TBUFFER_LOAD_FORMAT_XY_OFFEN,
                TBUFFER_LOAD_FORMAT_XY_IDXEN,
                TBUFFER_LOAD_FORMAT_XY_BOTHEN,
                TBUFFER_LOAD_FORMAT_XY_ADDR64,
                TBUFFER_LOAD_FORMAT_XYZ_OFFSET,
                TBUFFER_LOAD_FORMAT_XYZ_OFFEN,
                TBUFFER_LOAD_FORMAT_XYZ_IDXEN,
                TBUFFER_LOAD_FORMAT_XYZ_BOTHEN,
                TBUFFER_LOAD_FORMAT_XYZ_ADDR64,
                TBUFFER_LOAD_FORMAT_XYZW_OFFSET,
                TBUFFER_LOAD_FORMAT_XYZW_OFFEN,
                TBUFFER_LOAD_FORMAT_XYZW_IDXEN,
                TBUFFER_LOAD_FORMAT_XYZW_BOTHEN,
                TBUFFER_LOAD_FORMAT_XYZW_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 2: TBUFFER_LOAD_FORMAT_D16 — Typed buffer loads (D16 variants)
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 276-339.
//   vega-shader-instruction-set-architecture.pdf, pages 211-283.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 213-285.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 201-271.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 196-264.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 477-556.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 523-604.
//   rdna-shader-instruction-set-architecture.pdf, pages 230-287.
//   rdna2-shader-instruction-set-architecture.pdf, pages 224-281.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 91-564.
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_LOAD_FORMAT_D16_X_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_X_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_X_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_X_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_X_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XY_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XY_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XY_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XY_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XY_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XYZ_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XYZ_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XYZW_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XYZW_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 3: TBUFFER_STORE_FORMAT — Typed buffer stores (non-D16)
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 276-339.
//   vega-shader-instruction-set-architecture.pdf, pages 211-283.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 213-285.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 201-271.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 196-264.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 477-556.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 523-604.
//   rdna-shader-instruction-set-architecture.pdf, pages 230-287.
//   rdna2-shader-instruction-set-architecture.pdf, pages 224-281.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 91-564.
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_STORE_FORMAT_X_OFFSET,
                TBUFFER_STORE_FORMAT_X_OFFEN,
                TBUFFER_STORE_FORMAT_X_IDXEN,
                TBUFFER_STORE_FORMAT_X_BOTHEN,
                TBUFFER_STORE_FORMAT_X_ADDR64,
                TBUFFER_STORE_FORMAT_XY_OFFSET,
                TBUFFER_STORE_FORMAT_XY_OFFEN,
                TBUFFER_STORE_FORMAT_XY_IDXEN,
                TBUFFER_STORE_FORMAT_XY_BOTHEN,
                TBUFFER_STORE_FORMAT_XY_ADDR64,
                TBUFFER_STORE_FORMAT_XYZ_OFFSET,
                TBUFFER_STORE_FORMAT_XYZ_OFFEN,
                TBUFFER_STORE_FORMAT_XYZ_IDXEN,
                TBUFFER_STORE_FORMAT_XYZ_BOTHEN,
                TBUFFER_STORE_FORMAT_XYZ_ADDR64,
                TBUFFER_STORE_FORMAT_XYZW_OFFSET,
                TBUFFER_STORE_FORMAT_XYZW_OFFEN,
                TBUFFER_STORE_FORMAT_XYZW_IDXEN,
                TBUFFER_STORE_FORMAT_XYZW_BOTHEN,
                TBUFFER_STORE_FORMAT_XYZW_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 4: TBUFFER_STORE_FORMAT_D16 — Typed buffer stores (D16 variants)
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 276-339.
//   vega-shader-instruction-set-architecture.pdf, pages 211-283.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 213-285.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 201-271.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 196-264.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 477-556.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 523-604.
//   rdna-shader-instruction-set-architecture.pdf, pages 230-287.
//   rdna2-shader-instruction-set-architecture.pdf, pages 224-281.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 91-564.
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_STORE_FORMAT_D16_X_OFFSET,
                TBUFFER_STORE_FORMAT_D16_X_OFFEN,
                TBUFFER_STORE_FORMAT_D16_X_IDXEN,
                TBUFFER_STORE_FORMAT_D16_X_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_X_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XY_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XY_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XY_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XY_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XY_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XYZ_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XYZ_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XYZW_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XYZW_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 5: VBUFFER TBUFFER_LOAD_FORMAT — GFX12+ typed buffer loads (non-D16)
//
// Sources:
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_LOAD_FORMAT_X_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_X_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_X_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_X_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_X_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_XY_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_XY_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_XY_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_XY_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_XY_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_XYZ_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_XYZ_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_XYZ_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_XYZ_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_XYZ_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_XYZW_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_XYZW_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_XYZW_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_XYZW_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_XYZW_VBUFFER_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 6: VBUFFER TBUFFER_LOAD_FORMAT_D16 — GFX12+ typed buffer loads (D16)
//
// Sources:
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_LOAD_FORMAT_D16_X_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_X_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_X_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_X_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_X_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XY_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XY_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XY_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XY_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XY_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_ADDR64,
                TBUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_OFFSET,
                TBUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_OFFEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_IDXEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_BOTHEN,
                TBUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 7: VBUFFER TBUFFER_STORE_FORMAT — GFX12+ typed buffer stores (non-D16)
//
// Sources:
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_STORE_FORMAT_X_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_X_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_X_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_X_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_X_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_XY_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_XY_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_XY_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_XY_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_XY_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_XYZ_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_XYZ_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_XYZ_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_XYZ_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_XYZ_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_XYZW_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_XYZW_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_XYZW_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_XYZW_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_XYZW_VBUFFER_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// Section 8: VBUFFER TBUFFER_STORE_FORMAT_D16 — GFX12+ typed buffer stores (D16)
//
// Sources:
//   rdna4-instruction-set-architecture.pdf, pages 114-647.
//------------------------------------------------------------------------------

foreach inst = [TBUFFER_STORE_FORMAT_D16_X_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_D16_X_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_D16_X_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_D16_X_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_X_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XY_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XY_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XY_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XY_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XY_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_ADDR64,
                TBUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_OFFSET,
                TBUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_OFFEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_IDXEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_BOTHEN,
                TBUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_ADDR64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}
