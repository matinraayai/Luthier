//===- InstSISemantics.td -------------------------------------*- tblgen -*-==//
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
include "SISemOps.td"

class InstSISemantic<InstSI inst, list<dag> semantic = []> {
  InstSI Instruction = inst;
  list<dag> Semantic = semantic;
}

class InstSIIntrinsic<InstSI inst, Intrinsic intrinsic> {
  InstSI Instruction = inst;
  Intrinsic Intrinsic = intrinsic;
}

include "SOP1Sem.td"
include "SOP2Sem.td"
include "SOPCSem.td"
include "SOPKSem.td"
include "SOPPSem.td"
include "SMEMSem.td"
//include "MUBUFSem.td"
//include "MTBUFSem.td"
//include "FLATSem.td"
//include "VINTRPSem.td"
//include "VOPSem.td"
//include "VOP3Sem.td"
//include "VOP3CSem.td"
//include "VOP3PSem.td"
//include "VOPCSem.td"
//include "DSSem.td"