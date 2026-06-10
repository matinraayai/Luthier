//===-- DenseMapInfoTest.cpp ----------------------------------*- C++ -*-===//
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
/// \file
/// Unit tests for the \c DenseMapInfo specializations in
/// \c luthier/Common/DenseMapInfo.h.
//===----------------------------------------------------------------------===//
#include "luthier/Common/DenseMapInfo.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>

#include <deque>
#include <functional>

namespace {

using RefInt = std::reference_wrapper<int>;
using Info = llvm::DenseMapInfo<RefInt>;

//===----------------------------------------------------------------------===//
// Basic insertion / lookup
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, InsertAndLookup) {
  int A = 1, B = 2, C = 3;
  llvm::DenseMap<RefInt, llvm::StringRef> Map;
  Map[std::ref(A)] = "a";
  Map[std::ref(B)] = "b";
  Map[std::ref(C)] = "c";

  EXPECT_EQ(Map.size(), 3u);
  ASSERT_TRUE(Map.contains(std::ref(A)));
  EXPECT_EQ(Map.lookup(std::ref(A)), "a");
  EXPECT_EQ(Map.lookup(std::ref(B)), "b");
  EXPECT_EQ(Map.lookup(std::ref(C)), "c");

  // Re-assigning through a wrapper to the same object overwrites the entry.
  Map[std::ref(A)] = "a2";
  EXPECT_EQ(Map.size(), 3u);
  EXPECT_EQ(Map.lookup(std::ref(A)), "a2");
}

//===----------------------------------------------------------------------===//
// Identity, not value: keys compare by address
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, IdentityNotValue) {
  // Two *distinct* objects that happen to hold equal values must be distinct
  // keys; this is the core contract and guards against value-based hashing.
  int X = 5, Y = 5;
  ASSERT_NE(&X, &Y);

  llvm::DenseMap<RefInt, int> Map;
  Map[std::ref(X)] = 100;
  Map[std::ref(Y)] = 200;
  EXPECT_EQ(Map.size(), 2u);
  EXPECT_EQ(Map.lookup(std::ref(X)), 100);
  EXPECT_EQ(Map.lookup(std::ref(Y)), 200);

  // Two wrappers built independently around the *same* object collide.
  Map[std::ref(X)] = 101;
  EXPECT_EQ(Map.size(), 2u);
  EXPECT_EQ(Map.lookup(std::ref(X)), 101);

  // Mutating the referenced value does not move the key (address is stable).
  X = 999;
  EXPECT_TRUE(Map.contains(std::ref(X)));
  EXPECT_EQ(Map.lookup(std::ref(X)), 101);
}

//===----------------------------------------------------------------------===//
// DenseSet dedup
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, DenseSetDedup) {
  int A = 7, B = 7;
  llvm::DenseSet<RefInt> Set;
  EXPECT_TRUE(Set.insert(std::ref(A)).second);
  // Same object again -> not inserted.
  EXPECT_FALSE(Set.insert(std::ref(A)).second);
  // Different object, same value -> inserted.
  EXPECT_TRUE(Set.insert(std::ref(B)).second);
  EXPECT_EQ(Set.size(), 2u);
  EXPECT_TRUE(Set.contains(std::ref(A)));
  EXPECT_TRUE(Set.contains(std::ref(B)));
}

//===----------------------------------------------------------------------===//
// Erase exercises the tombstone key
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, EraseUsesTombstone) {
  // Stable storage so addresses don't change as the deque grows.
  std::deque<int> Storage;
  for (int I = 0; I < 16; ++I)
    Storage.push_back(I);

  llvm::DenseMap<RefInt, int> Map;
  for (int &Slot : Storage)
    Map[std::ref(Slot)] = Slot * 10;
  EXPECT_EQ(Map.size(), Storage.size());

  // Erase the even-indexed objects; this routes deleted slots through the
  // tombstone key and forces isEqual()/getHashValue() to round-trip correctly.
  for (size_t I = 0; I < Storage.size(); I += 2)
    EXPECT_EQ(Map.erase(std::ref(Storage[I])), 1u);
  EXPECT_EQ(Map.size(), Storage.size() / 2);

  // Survivors are still findable with the right values; erased ones are gone.
  for (size_t I = 0; I < Storage.size(); ++I) {
    if (I % 2 == 0)
      EXPECT_FALSE(Map.contains(std::ref(Storage[I]))) << I;
    else
      EXPECT_EQ(Map.lookup(std::ref(Storage[I])), static_cast<int>(I) * 10)
          << I;
  }

  // Re-inserting an erased key into a tombstoned slot works.
  Map[std::ref(Storage[0])] = -1;
  EXPECT_EQ(Map.lookup(std::ref(Storage[0])), -1);
}

//===----------------------------------------------------------------------===//
// Growth / rehash stress
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, GrowthAndRehash) {
  // A large number of distinct objects forces several rehashes; verify every
  // key survives and that sentinels never alias a real entry.
  constexpr int N = 1000;
  std::deque<int> Storage(N);
  llvm::DenseMap<RefInt, int> Map;
  for (int I = 0; I < N; ++I) {
    Storage[I] = I;
    Map[std::ref(Storage[I])] = I;
  }
  ASSERT_EQ(Map.size(), static_cast<size_t>(N));
  for (int I = 0; I < N; ++I) {
    ASSERT_TRUE(Map.contains(std::ref(Storage[I]))) << I;
    EXPECT_EQ(Map.lookup(std::ref(Storage[I])), I) << I;
  }
}

//===----------------------------------------------------------------------===//
// Const-qualified referent
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, ConstQualifiedKey) {
  // reference_wrapper<const int> threads const through DenseMapInfo<const
  // int*>.
  const int A = 1, B = 2;
  llvm::DenseMap<std::reference_wrapper<const int>, int> Map;
  Map[std::cref(A)] = 10;
  Map[std::cref(B)] = 20;
  EXPECT_EQ(Map.size(), 2u);
  EXPECT_EQ(Map.lookup(std::cref(A)), 10);
  EXPECT_EQ(Map.lookup(std::cref(B)), 20);
}

//===----------------------------------------------------------------------===//
// Non-trivial referent type (nothing int-specific in the specialization)
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, NonTrivialReferentType) {
  struct Widget {
    int Id;
    std::string Name;
  };
  Widget W1{1, "one"}, W2{2, "two"};
  llvm::DenseMap<std::reference_wrapper<Widget>, int> Map;
  Map[std::ref(W1)] = 1;
  Map[std::ref(W2)] = 2;
  EXPECT_EQ(Map.size(), 2u);
  EXPECT_EQ(Map.lookup(std::ref(W1)), 1);
  EXPECT_EQ(Map.lookup(std::ref(W2)), 2);
}

//===----------------------------------------------------------------------===//
// Trait functions directly
//===----------------------------------------------------------------------===//

TEST(DenseMapInfoTest, DirectTraitInvariants) {
  RefInt Empty = Info::getEmptyKey();

  // The sentinel is reflexively equal to itself.
  EXPECT_TRUE(Info::isEqual(Empty, Empty));

  // The sentinel address round-trips to the underlying T* sentinel, so it
  // never aliases a real object and hashes consistently with DenseMapInfo<T*>.
  EXPECT_EQ(&Empty.get(), llvm::DenseMapInfo<int *>::getEmptyKey());

  int A = 42;
  EXPECT_TRUE(Info::isEqual(std::ref(A), std::ref(A)));
  EXPECT_EQ(Info::getHashValue(std::ref(A)),
            llvm::DenseMapInfo<int *>::getHashValue(&A));

  // A real key is not equal to the sentinel.
  EXPECT_FALSE(Info::isEqual(std::ref(A), Empty));
}

} // namespace
