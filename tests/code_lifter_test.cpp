#include <luthier/luthier.h>
#include "code_lifter.hpp"

#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(CodeLifter, BasicAssertions) {
  auto & CodeLifter = luthier::CodeLifter::instance();
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

void luthier::atToolInit(luthier::ApiEvtPhase Phase) {};

void luthier::atFinalization(luthier::ApiEvtPhase Phase) {}
