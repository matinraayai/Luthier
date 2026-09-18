/**
 * @file audit_hook_dynamic.h
 * @brief C/C++ definitions for dynamic dispatch and caller filtering[cite: 16].
 * 
 * Provides the enums and functions required to dynamically route hooks
 * and configure filter states (include/exclude) based on caller origin[cite: 16].
 */

#ifndef AUDIT_HOOK_DYNAMIC_H
#define AUDIT_HOOK_DYNAMIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Defines the operational mode for caller filtering[cite: 16].
 */
typedef enum {
  AH_FILTER_GLOBAL = 0, ///< The hook applies to all callers globally[cite: 16].
  AH_FILTER_INCLUDE,    ///< The hook applies strictly to the specified libraries[cite: 16].
  AH_FILTER_EXCLUDE     ///< The hook applies to all callers except the specified libraries[cite: 16].
} ah_filter_mode_t;

/**
 * @brief Error codes returned by dynamic hook operations[cite: 16].
 */
typedef enum {
  AH_SUCCESS = 0,               ///< Operation completed successfully[cite: 16].
  AH_ERR_NOT_HOOKED = -1,       ///< The target symbol is not currently hooked by this tool[cite: 16].
  AH_ERR_NOT_DYNAMIC = -2,      ///< The hook chain lacks the dynamic dispatcher[cite: 16].
  AH_ERR_NOT_GLOBAL = -3,       ///< Cannot safely retrieve target; hook has active caller filters[cite: 16].
  AH_ERR_MULTIPLE_RULES = -4    ///< Cannot safely retrieve target; multiple active inclusion/exclusion rules apply[cite: 16].
} ah_err_t;

/**
 * @brief Configures caller filtering rules for a specific tool's hook[cite: 16].
 *
 * Maintains persistent include/exclude lists per hook across subsequent calls[cite: 16].
 * New registrations are treated as set operations on the current state[cite: 16].
 *
 * @param tool_name The unique string identifier for the plugin registering the filter[cite: 16].
 * @param mode The filtering mode to apply (Global, Include, or Exclude)[cite: 16].
 * @param libs Array of C-strings specifying library names to filter against[cite: 16].
 * @param num_libs The number of libraries in the `libs` array[cite: 16].
 * @return 0 on success, or a non-zero error code if configuration fails[cite: 16].
 */
int ah_set_caller_filter(const char *tool_name, ah_filter_mode_t mode,
                         const char **libs, size_t num_libs);

/**
 * @brief Dynamically updates the target trampoline for a registered hook[cite: 16].
 * 
 * Used to securely bridge namespace bounds and override hook execution 
 * directly from the target application[cite: 16].
 *
 * @param tool_name The unique string identifier for the plugin[cite: 16].
 * @param new_target A pointer to the new function target[cite: 16].
 * @return 0 on success, or a non-zero error code[cite: 16].
 */
int ah_set_target(const char *tool_name, void *new_target);

/**
 * @brief Retrieves the current target of a global dynamic dispatch hook[cite: 16].
 * 
 * @param tool_name The unique string identifier for the plugin[cite: 16].
 * @param out_target A double pointer where the current target address will be stored[cite: 16].
 * @return AH_SUCCESS and sets out_target if the hook is strictly global[cite: 16].
 *         Returns specific ah_err_t codes if the constraints are violated[cite: 16].
 */
ah_err_t ah_get_target(const char *tool_name, void **out_target);

#ifdef __cplusplus
} /* extern "C" */

/* --- Modern C++ Overloads --- */
#include <concepts>
#include <ranges>
#include <vector>

/**
 * @brief Modern C++ overload for configuring caller filtering rules[cite: 16].
 *
 * Evaluates the provided C++ range and seamlessly forwards it to the core C API[cite: 16].
 *
 * @tparam Range A type satisfying std::ranges::forward_range whose elements are convertible to const char*[cite: 16].
 * @param tool_name The unique identifier for the plugin[cite: 16].
 * @param mode The filtering mode to apply[cite: 16].
 * @param libs The range of library names (e.g., std::vector<const char*>)[cite: 16].
 * @return 0 on success, or a non-zero error code[cite: 16].
 */
template <std::ranges::forward_range Range>
  requires std::convertible_to<std::ranges::range_value_t<Range>, const char *>
inline int ah_set_caller_filter(const char *tool_name, ah_filter_mode_t mode,
                                const Range &libs) {
  std::vector<const char *> contiguous_libs;

  if constexpr (std::ranges::sized_range<Range>) {
    contiguous_libs.reserve(std::ranges::size(libs));
  }

  for (const auto &lib : libs) {
    contiguous_libs.push_back(lib);
  }

  return ah_set_caller_filter(tool_name, mode, contiguous_libs.data(),
                              contiguous_libs.size());
}

#endif /* __cplusplus */

#endif /* AUDIT_HOOK_DYNAMIC_H */