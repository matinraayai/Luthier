//===-- ToolRegionTracker.h --------------------------------------*- C++
//-*-===//
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
/// Defines the \c ToolRegionTrackerTrait, its type-erased
/// version \c ToolRegionTracker for passing around to other classes, as well
/// as \c ToolRegionMarker for RAII marking of tool regions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_TOOL_REGION_MARKER_H
#define LUTHIER_TOOLING_TOOL_REGION_MARKER_H
#include <type_traits>

namespace luthier {

class ToolRegionTracker;

/// \brief A trait for tracking calls performed by the tool to an underlying
/// application on the current thread. Useful for preventing re-interception
/// of tool calls to GPU runtimes (e.g. HSA)
template <typename> class ToolRegionTrackerTrait {

  static unsigned int thread_local ToolRegionDepth;

  /// \brief Marks the beginning of the tool code region on the caller thread,
  /// if not already marked as such. Calling this function multiple times is
  /// safe and will increment the thread's tool region depth counter internally.
  static void beginToolRegion() { ToolRegionDepth++; }

  /// \brief Marks the end of the tool code region. Safe to call multiple times.
  static void endToolRegion() {
    if (ToolRegionDepth == 0) {
      return;
    }
    ToolRegionDepth--;
  }

  /// \brief Whether the current region belongs to the tool on the current
  /// thread.
  [[nodiscard]] bool insideToolRegion() { return ToolRegionDepth != 0; }
};

/// \brief Non-owning, type-erased handle to a type implementing
/// \c ToolRegionTrackerTrait.
class ToolRegionTracker {
  /// \brief The erased interface of \c ToolRegionTrackerTrait.
  struct Concept {
    virtual ~Concept() = default;

    virtual void beginToolRegion() = 0;

    virtual void endToolRegion() = 0;

    [[nodiscard]] virtual bool insideToolRegion() const = 0;
  };

  /// \brief The one polymorphic type in this design, generated per marker.
  template <typename T> struct Model final : Concept {
    T &M;

    explicit Model(T &Marker) : M(Marker) {}

    void beginToolRegion() override {
      ToolRegionTrackerTrait<T>::beginToolRegion();
    }

    void endToolRegion() override {
      ToolRegionTrackerTrait<T>::endToolRegion();
    }

    [[nodiscard]] bool insideToolRegion() const override {
      return ToolRegionTrackerTrait<T>::insideToolRegion();
    }
  };

  Concept &Impl;

public:
  /// \brief Construct a handle referring to \p Marker.
  template <typename T,
            std::enable_if_t<
                !std::is_same_v<std::decay_t<T>, ToolRegionTracker>, int> = 0>
  /*implicit*/ ToolRegionTracker(T &Marker) : Impl(Marker) {}

  /// \copydoc ToolRegionTrackerTrait<>::beginToolRegion
  void beginToolRegion() { Impl.beginToolRegion(); }

  /// \copydoc ToolRegionTrackerTrait<>::endToolRegion
  void endToolRegion() { Impl.endToolRegion(); }

  /// \copydoc ToolRegionTrackerTrait<>::insideToolRegion
  [[nodiscard]] bool insideToolRegion() const {
    return Impl.insideToolRegion();
  }
};

/// \brief A RAII class to safely mark the code region in a thread as the
/// tool's.
class ToolRegionMarker {
  ToolRegionTracker &T;

public:
  explicit ToolRegionMarker(ToolRegionTracker &Tracker) : T(Tracker) {
    T.beginToolRegion();
  }

  ToolRegionMarker(const ToolRegionMarker &) = delete;
  ToolRegionMarker &operator=(const ToolRegionMarker &) = delete;

  ~ToolRegionMarker() { T.endToolRegion(); }
};

} // namespace luthier

#endif
