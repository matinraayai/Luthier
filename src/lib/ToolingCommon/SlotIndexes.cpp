//===-- SlotIndexes.cpp - Slot Indexes  ------------------------------===//

#include "luthier/Tooling/SlotIndexes.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

using namespace luthier;

#define DEBUG_TYPE "slotindexes"

SlotIndexes::~SlotIndexes() {
  // The indexList's nodes are all allocated in the BumpPtrAllocator.
  indexList.clear();
}
void SlotIndexes::clear() {
  mi2iMap.clear();
  MBBRanges.clear();
  idx2MBBMap.clear();
  indexList.clear();
  ileAllocator.Reset();
}

void SlotIndexes::analyze(PredicatedMachineFunction &PMF) {

  // Compute numbering as follows:
  // Grab an iterator to the start of the index list.
  // Iterate over all MBBs, and within each MBB all MIs, keeping the MI
  // iterator in lock-step (though skipping it over indexes which have
  // null pointers in the instruction field).
  // At each iteration assert that the instruction pointed to in the index
  // is the same one pointed to by the MI iterator. This

  // FIXME: This can be simplified. The mi2iMap_, Idx2MBBMap, etc. should
  // only need to be set up once after the first numbering is computed.

  mf = &PMF;

  // Check that the list contains only the sentinel.
  assert(indexList.empty() && "Index list non-empty at initial numbering?");
  assert(idx2MBBMap.empty() &&
         "Index -> MBB mapping non-empty at initial numbering?");
  assert(MBBRanges.empty() &&
         "MBB -> Index mapping non-empty at initial numbering?");
  assert(mi2iMap.empty() &&
         "MachineInstr -> Index mapping non-empty at initial numbering?");

  unsigned index = 0;
  MBBRanges.resize(PMF.getParent().getNumVecMBBs());
  idx2MBBMap.reserve(PMF.getParent().getNumVecMBBs());

  indexList.push_back(*createEntry(nullptr, index));

  // Iterate over the function, we get the assosciated LinearMachineBasicBlock and iterate over PredicatedMachineBasicBlock.
  for (LinearMachineBasicBlock &MBB : PMF) {
    for(PredicatedMachineBasicBlock& PMBB : MBB){

        // Insert an index for the MBB start.
        SlotIndex blockStartIndex(&indexList.back(), SlotIndex::Slot_Block);

        for (llvm::MachineInstr &MI : PMBB) {
        if (MI.isDebugOrPseudoInstr())
            continue;

        // Insert a store index for the instr.
        indexList.push_back(*createEntry(&MI, index += SlotIndex::InstrDist));

        // Save this base index in the maps.
        mi2iMap.insert(std::make_pair(
            &MI, SlotIndex(&indexList.back(), SlotIndex::Slot_Block)));
        }

        // We insert one blank instructions between basic blocks.
        indexList.push_back(*createEntry(nullptr, index += SlotIndex::InstrDist));

        MBBRanges[PMBB.getGlobalNumber()].first = blockStartIndex;
        MBBRanges[PMBB.getGlobalNumber()].second = SlotIndex(&indexList.back(),
                                                    SlotIndex::Slot_Block);
        idx2MBBMap.push_back(IdxMBBPair(blockStartIndex, &PMBB));
    }
  }

  // Sort the Idx2MBBMap
  llvm::sort(idx2MBBMap, less_first());

  LLVM_DEBUG(mf->print(dbgs(), this));
}

void SlotIndexes::removeMachineInstrFromMaps(llvm::MachineInstr &MI,
                                             bool AllowBundled) {
  assert((AllowBundled || !MI.isBundledWithPred()) &&
         "Use removeSingleMachineInstrFromMaps() instead");
  Mi2IndexMap::iterator mi2iItr = mi2iMap.find(&MI);
  if (mi2iItr == mi2iMap.end())
    return;

  SlotIndex MIIndex = mi2iItr->second;
  IndexListEntry &MIEntry = *MIIndex.listEntry();
  assert(MIEntry.getInstr() == &MI && "Instruction indexes broken.");
  mi2iMap.erase(mi2iItr);
  // FIXME: Eventually we want to actually delete these indexes.
  MIEntry.setInstr(nullptr);
}

// FIXME: Do we need a vectorMBB iterator?
void SlotIndexes::removeSingleMachineInstrFromMaps(llvm::MachineInstr &MI) {
  Mi2IndexMap::iterator mi2iItr = mi2iMap.find(&MI);
  if (mi2iItr == mi2iMap.end())
    return;

  SlotIndex MIIndex = mi2iItr->second;
  IndexListEntry &MIEntry = *MIIndex.listEntry();
  assert(MIEntry.getInstr() == &MI && "Instruction indexes broken.");
  mi2iMap.erase(mi2iItr);

  // When removing the first instruction of a bundle update mapping to next
  // instruction.
  if (MI.isBundledWithSucc()) {
    // Only the first instruction of a bundle should have an index assigned.
    assert(!MI.isBundledWithPred() && "Should be first bundle instruction");

    llvm::MachineBasicBlock::instr_iterator Next = std::next(MI.getIterator());
    llvm::MachineInstr &NextMI = *Next;
    MIEntry.setInstr(&NextMI);
    mi2iMap.insert(std::make_pair(&NextMI, MIIndex));
    return;
  } else {
    // FIXME: Eventually we want to actually delete these indexes.
    MIEntry.setInstr(nullptr);
  }
}

// Renumber indexes locally after curItr was inserted, but failed to get a new
// index.
void SlotIndexes::renumberIndexes(IndexList::iterator curItr) {
  // Number indexes with half the default spacing so we can catch up quickly.
  const unsigned Space = SlotIndex::InstrDist/2;
  static_assert((Space & 3) == 0, "InstrDist must be a multiple of 2*NUM");

  IndexList::iterator startItr = std::prev(curItr);
  unsigned index = startItr->getIndex();
  do {
    curItr->setIndex(index += Space);
    ++curItr;
    // If the next index is bigger, we have caught up.
  } while (curItr != indexList.end() && curItr->getIndex() <= index);

  LLVM_DEBUG(llvm::dbgs() << "\n*** Renumbered SlotIndexes " << startItr->getIndex()
                    << '-' << index << " ***\n");
  ++NumLocalRenum;
}

// Repair indexes after adding and removing instructions.
void SlotIndexes::repairIndexesInRange(PredicatedMachineBasicBlock *MBB,
                                       PredicatedMachineBasicBlock::iterator Begin,
                                       PredicatedMachineBasicBlock::iterator End) {
  bool includeStart = (Begin == MBB->begin());
  SlotIndex startIdx;
  if (includeStart)
    startIdx = getMBBStartIdx(MBB);
  else
    startIdx = getInstructionIndex(*--Begin);

  SlotIndex endIdx;
  if (End == MBB->end())
    endIdx = getMBBEndIdx(MBB);
  else
    endIdx = getInstructionIndex(*End);

  // FIXME: Conceptually, this code is implementing an iterator on MBB that
  // optionally includes an additional position prior to MBB->begin(), indicated
  // by the includeStart flag. This is done so that we can iterate MIs in a MBB
  // in parallel with SlotIndexes, but there should be a better way to do this.
  IndexList::iterator ListB = startIdx.listEntry()->getIterator();
  IndexList::iterator ListI = endIdx.listEntry()->getIterator();
  PredicatedMachineBasicBlock::iterator MBBI = End;
  bool pastStart = false;
  bool OldIndexesRemoved = false;
  while (ListI != ListB || MBBI != Begin || (includeStart && !pastStart)) {
    assert(ListI->getIndex() >= startIdx.getIndex() &&
           (includeStart || !pastStart) &&
           "Decremented past the beginning of region to repair.");

    llvm::MachineInstr *SlotMI = ListI->getInstr();
    llvm::MachineInstr *MI = (MBBI != MBB->end() && !pastStart) ? &*MBBI : nullptr;
    bool MBBIAtBegin = MBBI == Begin && (!includeStart || pastStart);
    bool MIIndexNotFound = MI && !mi2iMap.contains(MI);
    bool SlotMIRemoved = false;

    if (SlotMI == MI && !MBBIAtBegin) {
      --ListI;
      if (MBBI != Begin)
        --MBBI;
      else
        pastStart = true;
    } else if (MIIndexNotFound || OldIndexesRemoved) {
      if (MBBI != Begin)
        --MBBI;
      else
        pastStart = true;
    } else {
      // We ran through all the indexes on the interval
      //   -> The only thing left is to go through all the
      //   remaining MBB instructions and update their indexes
      if (ListI == ListB)
        OldIndexesRemoved = true;
      else
        --ListI;
      if (SlotMI) {
        removeMachineInstrFromMaps(*SlotMI);
        SlotMIRemoved = true;
      }
    }

    llvm::MachineInstr *InstrToInsert = SlotMIRemoved ? SlotMI : MI;

    // Insert instruction back into the maps after passing it/removing the index
    if ((MIIndexNotFound || SlotMIRemoved) && InstrToInsert->getParent() &&
        !InstrToInsert->isDebugOrPseudoInstr())
      insertMachineInstrInMaps(*InstrToInsert);
  }
}

void SlotIndexes::packIndexes() {
  for (auto [Index, Entry] : enumerate(indexList))
    Entry.setIndex(Index * SlotIndex::InstrDist);
}

void SlotIndexes::print(llvm::raw_ostream &OS) const {
  for (const IndexListEntry &ILE : indexList) {
    OS << ILE.getIndex() << ' ';

    if (ILE.getInstr())
      OS << *ILE.getInstr();
    else
      OS << '\n';
  }

  for (unsigned i = 0, e = MBBRanges.size(); i != e; ++i)
    OS << "%bb." << i << "\t[" << MBBRanges[i].first << ';'
       << MBBRanges[i].second << ")\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SlotIndexes::dump() const { print(llvm::dbgs()); }
#endif

// Print a SlotIndex to a raw_ostream.
void SlotIndex::print(llvm::raw_ostream &os) const {
  if (isValid())
    os << listEntry()->getIndex() << "Berd"[getSlot()];
  else
    os << "invalid";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// Dump a SlotIndex to stderr.
LLVM_DUMP_METHOD void SlotIndex::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}
#endif
