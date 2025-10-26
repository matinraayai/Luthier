//===-- Statistics.cpp ----------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "klee/Statistics/Statistics.h"

#include <utility>
#include <vector>

using namespace luthier::klee;

StatisticManager::StatisticManager()
    : Enabled(true), GlobalStats(nullptr), indexedStats(nullptr),
      contextStats(nullptr), index(0) {}

StatisticManager::~StatisticManager() {
  delete[] GlobalStats;
  delete[] indexedStats;
}

void StatisticManager::useIndexedStats(unsigned totalIndices) {
  delete[] indexedStats;
  indexedStats = new uint64_t[totalIndices * stats.size()];
  memset(indexedStats, 0, sizeof(*indexedStats) * totalIndices * stats.size());
}

void StatisticManager::registerStatistic(Statistic &s) {
  delete[] GlobalStats;
  s.Id = Stats.size();
  Stats.push_back(&s);
  GlobalStats = new uint64_t[Stats.size()];
  memset(GlobalStats, 0, sizeof(*GlobalStats) * Stats.size());
}

int StatisticManager::getStatisticID(const std::string &name) const {
  for (unsigned i = 0; i < stats.size(); i++)
    if (stats[i]->getName() == name)
      return i;
  return -1;
}

Statistic *StatisticManager::getStatisticByName(const std::string &name) const {
  for (unsigned i = 0; i < stats.size(); i++)
    if (stats[i]->getName() == name)
      return stats[i];
  return 0;
}

StatisticManager *klee::theStatisticManager = 0;

static StatisticManager &getStatisticManager() {
  static StatisticManager sm;
  theStatisticManager = &sm;
  return sm;
}

/* *** */

Statistic::Statistic(StatisticManager &Manager, std::string Name,
                     std::string ShortName)
    : Name{std::move(Name)}, ShortName{std::move(ShortName)}, Manager(Manager) {
  Manager.registerStatistic(*this);
}

Statistic &Statistic::operator+=(std::uint64_t addend) {
  Manager.incrementStatistic(*this, addend);
  return *this;
}

std::uint64_t Statistic::getValue() const { return Manager.getValue(*this); }
