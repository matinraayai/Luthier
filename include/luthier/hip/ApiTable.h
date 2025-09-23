
#ifndef LUTHIER_HIP_API_TABLE_H
#define LUTHIER_HIP_API_TABLE_H
#include <hip/amd_detail/hip_api_trace.hpp>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hip {

template <auto Entry> bool apiTableHasEntry(const auto &Table) {
  return static_cast<size_t>(&(Table.*Entry)) < Table.size;
}

template <rocprofiler_intercept_table_t TableType> struct ApiTableEnumInfo;

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
};

template <rocprofiler_intercept_table_t TableType> class ApiTableContainer {
private:
  const typename ApiTableEnumInfo<TableType>::ApiTableType &ApiTable{};

public:
  explicit ApiTableContainer(
      const ApiTableEnumInfo<TableType>::ApiTableType &ApiTable)
      : ApiTable(ApiTable) {};

  /// \brief Checks if the \c Func is present in the API table snapshot
  /// \tparam Func pointer-to-member of the function entry inside the
  /// extension table being queried
  /// \return \c true if the function is available inside the
  /// API table, \c false otherwise. Reports a fatal error
  /// if the snapshot has not been initialized by rocprofiler-sdk
  template <auto ApiTableEnumInfo<TableType>::*Func>
  [[nodiscard]] bool tableSupportsFunction() const {
    return apiTableHasEntry<Func>(ApiTable);
  }

  /// \returns the function inside the snapshot associated with the
  /// pointer-to-member accessor \c Func
  template <auto ApiTableEnumInfo<TableType>::*Func>
  const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        tableSupportsFunction<Func>(),
        "The passed function is not inside the table."));
    return *(ApiTable.*Func);
  }

  /// Obtains the function \c Func from the table snapshot and calls
  /// it with the passed \p Args and returns the results of the function call
  template <auto ApiTableEnumInfo<TableType>::*Func, typename... ArgTypes>
  auto callFunction(ArgTypes... Args) const {
    return getFunction<Func>()(Args...);
  }
};

} // namespace luthier::hip

#endif