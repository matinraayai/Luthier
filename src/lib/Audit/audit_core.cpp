#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "audit_hook.hpp"
#include <algorithm>
#include <cstdarg>
#include <dlfcn.h>
#include <link.h>
#include <mutex>
#include <shared_mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Diagnostics & Logging
// -----------------------------------------------------------------------------

static bool ah_debug_enabled = false;

inline void ah_debug_log(const char *format, ...) {
  if (ah_debug_enabled) {
    fprintf(stderr, "[AH_DEBUG] ");
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
  }
}

// -----------------------------------------------------------------------------
// State Management & Dynamic Data Structures
// -----------------------------------------------------------------------------

struct hook_action_t {
  std::string tool_name;
  void *trampoline_ptr;
  void **original_out_ptr;
  void *dispatcher_ptr;
  ah_filter_mode_t filter_mode;
  std::vector<std::string> filter_libs;
};

struct symbol_chain_t {
  std::string symbol_name;
  std::vector<hook_action_t> actions;
  void *native_os_ptr = nullptr;
  bool is_dynamic_dispatch = false;
  bool force_dispatcher_bind = false;
};

static std::unordered_map<std::string, symbol_chain_t> *get_hooks() {
  static auto *hooks = new std::unordered_map<std::string, symbol_chain_t>();
  return hooks;
}

static std::shared_mutex *get_hooks_mutex() {
  static auto *mutex = new std::shared_mutex();
  return mutex;
}

static std::unordered_map<uintptr_t, std::string> *get_objects() {
  static auto *objs = new std::unordered_map<uintptr_t, std::string>();
  return objs;
}

static std::shared_mutex *get_objects_mutex() {
  static auto *mutex = new std::shared_mutex();
  return mutex;
}

static thread_local bool tls_hooks_paused = false;
static thread_local bool tls_hooks_ignored = false;
static thread_local std::vector<void **> tls_active_orig_ptrs;

static thread_local std::unordered_map<std::string, std::string>
    tls_chain_callers;

// Callback registry for plugins that want la_objopen events
typedef void (*ah_plugin_on_objopen_t)(const char *, uintptr_t);
static std::vector<ah_plugin_on_objopen_t> plugin_objopen_callbacks;

static bool is_action_allowed(const hook_action_t &act,
                              const std::string &caller) {
  if (act.filter_mode == AH_FILTER_GLOBAL) {
    ah_debug_log("is_action_allowed: '%s' -> GLOBAL => ALLOWED\n",
                 act.tool_name.c_str());
    return true;
  }

  bool found = false;
  for (const auto &lib : act.filter_libs) {
    if (caller.find(lib) != std::string::npos) {
      found = true;
      break;
    }
  }

  bool allowed = false;
  if (act.filter_mode == AH_FILTER_INCLUDE)
    allowed = found;
  else if (act.filter_mode == AH_FILTER_EXCLUDE)
    allowed = !found;

  ah_debug_log("is_action_allowed: '%s' evaluating caller '%s' against %s "
               "filter => %s\n",
               act.tool_name.c_str(), caller.c_str(),
               (act.filter_mode == AH_FILTER_INCLUDE ? "INCLUDE" : "EXCLUDE"),
               (allowed ? "ALLOWED" : "DENIED"));

  return allowed;
}

// -----------------------------------------------------------------------------
// Core API
// -----------------------------------------------------------------------------

extern "C" {

int ah_register_hook(const char *tool_name, const char *symbol_name,
                     void *hook_func, void **original_out, void *dispatcher,
                     bool force_dynamic) {
  if (!tool_name || !symbol_name || !hook_func)
    return -1;

  std::unique_lock lock(*get_hooks_mutex());
  auto &chain = (*get_hooks())[symbol_name];
  chain.symbol_name = symbol_name;

  if (force_dynamic) {
    chain.is_dynamic_dispatch = true;
    chain.force_dispatcher_bind = true;
  }

  hook_action_t action;
  action.tool_name = tool_name;
  action.trampoline_ptr = hook_func;
  action.original_out_ptr = original_out;
  action.dispatcher_ptr = dispatcher;
  action.filter_mode = AH_FILTER_GLOBAL;

  bool is_new_replace = (original_out == nullptr);

  if (!chain.actions.empty()) {
    bool is_old_replace = (chain.actions.back().original_out_ptr == nullptr);

    if (is_new_replace && is_old_replace) {
      fprintf(stderr,
              "[AuditCore] WARNING: Symbol '%s' was already replaced by '%s', "
              "but is now being replaced again by '%s'. The previous "
              "replacement is discarded.\n",
              symbol_name, chain.actions.back().tool_name.c_str(), tool_name);
      chain.actions.back() = action;
      return 0;
    } else if (is_new_replace && !is_old_replace) {
      fprintf(stderr,
              "[AuditCore] WARNING: Symbol '%s' was wrapped by '%s', but is "
              "now being entirely replaced by '%s'. The previous wrapping is "
              "discarded.\n",
              symbol_name, chain.actions.back().tool_name.c_str(), tool_name);
      chain.actions.clear();
      chain.is_dynamic_dispatch = false;
      chain.force_dispatcher_bind = false;
      if (force_dynamic) {
        chain.is_dynamic_dispatch = true;
        chain.force_dispatcher_bind = true;
      }
      chain.actions.push_back(action);
      return 0;
    }
  }

  chain.actions.push_back(action);

  if (chain.actions.size() > 1 && !chain.is_dynamic_dispatch) {
    if (action.original_out_ptr) {
      *(action.original_out_ptr) =
          chain.actions[chain.actions.size() - 2].trampoline_ptr;
    }
  } else if (chain.is_dynamic_dispatch) {
    if (action.original_out_ptr && action.dispatcher_ptr) {
      *(action.original_out_ptr) = action.dispatcher_ptr;
    }
  }

  return 0;
}

int ah_set_caller_filter(const char *tool_name, ah_filter_mode_t mode,
                         const char **libs, size_t num_libs) {
  if (!tool_name || !libs)
    return -1;

  std::unique_lock lock(*get_hooks_mutex());

  for (auto &pair : *get_hooks()) {
    auto &chain = pair.second;
    bool modified = false;

    for (auto &act : chain.actions) {
      if (act.tool_name == tool_name) {
        modified = true;

        if (mode == AH_FILTER_INCLUDE) {
          if (act.filter_mode == AH_FILTER_EXCLUDE) {
            for (size_t i = 0; i < num_libs; ++i) {
              if (libs[i]) {
                std::string target = libs[i];
                auto it = std::find(act.filter_libs.begin(),
                                    act.filter_libs.end(), target);
                if (it != act.filter_libs.end()) {
                  act.filter_libs.erase(it);
                } else {
                  fprintf(stderr,
                          "[AuditCore] WARNING: Tool '%s' attempted to include "
                          "library '%s' on symbol '%s', but it was not in the "
                          "active EXCLUDE list.\n",
                          tool_name, target.c_str(), chain.symbol_name.c_str());
                }
              }
            }
          } else {
            if (act.filter_mode == AH_FILTER_GLOBAL) {
              act.filter_mode = AH_FILTER_INCLUDE;
              act.filter_libs.clear();
            }
            for (size_t i = 0; i < num_libs; ++i) {
              if (libs[i]) {
                std::string new_lib = libs[i];
                if (std::find(act.filter_libs.begin(), act.filter_libs.end(),
                              new_lib) == act.filter_libs.end()) {
                  act.filter_libs.push_back(new_lib);
                }
              }
            }
          }
        } else if (mode == AH_FILTER_EXCLUDE) {
          if (act.filter_mode == AH_FILTER_GLOBAL ||
              act.filter_mode == AH_FILTER_EXCLUDE) {
            act.filter_mode = AH_FILTER_EXCLUDE;
            for (size_t i = 0; i < num_libs; ++i) {
              if (libs[i]) {
                std::string new_lib = libs[i];
                if (std::find(act.filter_libs.begin(), act.filter_libs.end(),
                              new_lib) == act.filter_libs.end()) {
                  act.filter_libs.push_back(new_lib);
                }
              }
            }
          } else if (act.filter_mode == AH_FILTER_INCLUDE) {
            for (size_t i = 0; i < num_libs; ++i) {
              if (libs[i]) {
                std::string target = libs[i];
                auto it = std::find(act.filter_libs.begin(),
                                    act.filter_libs.end(), target);
                if (it != act.filter_libs.end()) {
                  act.filter_libs.erase(it);
                } else {
                  fprintf(stderr,
                          "[AuditCore] WARNING: Tool '%s' attempted to exclude "
                          "library '%s' on symbol '%s', but it was not in the "
                          "active INCLUDE list.\n",
                          tool_name, target.c_str(), chain.symbol_name.c_str());
                }
              }
            }
          }
        }
      }
    }

    if (modified && chain.actions.size() > 1 && !chain.is_dynamic_dispatch) {
      fprintf(stderr,
              "\n======================================================\n");
      fprintf(stderr,
              "[AuditCore] WARNING: Diverging Chain Detected on '%s'!\n",
              chain.symbol_name.c_str());
      fprintf(stderr,
              "[AuditCore] Applying caller filters to a multiply-hooked\n");
      fprintf(stderr,
              "[AuditCore] function forces Dynamic Dispatch routing.\n");
      fprintf(stderr,
              "======================================================\n\n");

      chain.is_dynamic_dispatch = true;
      for (auto &a : chain.actions) {
        if (a.original_out_ptr && a.dispatcher_ptr) {
          *(a.original_out_ptr) = a.dispatcher_ptr;
        }
      }
    }
  }
  return 0;
}

int ah_set_target(const char *tool_name, void *new_target) {
  if (!tool_name || !new_target)
    return -1;

  std::unique_lock lock(*get_hooks_mutex());
  for (auto &pair : *get_hooks()) {
    auto &chain = pair.second;
    for (auto &act : chain.actions) {
      if (act.tool_name == tool_name) {
        if (!chain.force_dispatcher_bind) {
          return -1;
        }
        act.filter_mode = AH_FILTER_GLOBAL;
        act.filter_libs.clear();
        act.trampoline_ptr = new_target;
        return 0;
      }
    }
  }
  return -1;
}

ah_err_t ah_get_target(const char *tool_name, void **out_target) {
  if (!tool_name || !out_target)
    return AH_ERR_NOT_HOOKED;

  std::shared_lock lock(*get_hooks_mutex());
  for (const auto &pair : *get_hooks()) {
    const auto &chain = pair.second;
    for (const auto &act : chain.actions) {
      if (act.tool_name == tool_name) {
        if (!chain.force_dispatcher_bind) {
          return AH_ERR_NOT_DYNAMIC;
        }

        if (act.filter_libs.size() > 1) {
          return AH_ERR_MULTIPLE_RULES;
        }

        if (act.filter_mode != AH_FILTER_GLOBAL) {
          return AH_ERR_NOT_GLOBAL;
        }

        *out_target = act.trampoline_ptr;
        return AH_SUCCESS;
      }
    }
  }

  return AH_ERR_NOT_HOOKED;
}

void ah_thread_pause_hooks(void) { tls_hooks_paused = true; }
void ah_thread_resume_hooks(void) { tls_hooks_paused = false; }
void ah_thread_ignore_hooks(void) { tls_hooks_ignored = true; }
bool ah_are_hooks_paused(void) { return tls_hooks_paused || tls_hooks_ignored; }

void ah_push_orig_ptr(void **orig_out) {
  tls_active_orig_ptrs.push_back(orig_out);
}
void ah_pop_orig_ptr(void) {
  if (!tls_active_orig_ptrs.empty())
    tls_active_orig_ptrs.pop_back();
}

void *ah_get_next_hop(void **orig_out, void *return_addr) {
  bool is_stepping_down = (!tls_active_orig_ptrs.empty() &&
                           tls_active_orig_ptrs.back() == orig_out);

  std::string caller_lib;
  std::string sym_name;

  std::shared_lock lock(*get_hooks_mutex());
  for (const auto &pair : *get_hooks()) {
    for (const auto &act : pair.second.actions) {
      if (act.original_out_ptr == orig_out) {
        sym_name = pair.second.symbol_name;
        break;
      }
    }
    if (!sym_name.empty())
      break;
  }

  if (!is_stepping_down) {
    Dl_info info;
    if (dladdr(return_addr, &info) && info.dli_fname) {
      caller_lib = info.dli_fname;
    } else {
      caller_lib = program_invocation_short_name;
    }
    if (!sym_name.empty()) {
      tls_chain_callers[sym_name] = caller_lib;
    }
  } else {
    if (!sym_name.empty()) {
      caller_lib = tls_chain_callers[sym_name];
    }
    if (caller_lib.empty()) {
      Dl_info info;
      if (dladdr(return_addr, &info) && info.dli_fname) {
        caller_lib = info.dli_fname;
      } else {
        caller_lib = program_invocation_short_name;
      }
    }
  }

  ah_debug_log(
      "ah_get_next_hop: orig_out=%p, caller='%s', is_stepping_down=%s\n",
      orig_out, caller_lib.c_str(), is_stepping_down ? "true" : "false");

  for (const auto &pair : *get_hooks()) {
    const auto &chain = pair.second;
    if (!chain.is_dynamic_dispatch)
      continue;

    for (int i = chain.actions.size() - 1; i >= 0; --i) {
      if (chain.actions[i].original_out_ptr == orig_out) {
        int start_idx = is_stepping_down ? i - 1 : i;

        ah_debug_log("ah_get_next_hop: Found tool '%s' matched to orig_out. "
                     "Evaluating from chain index %d down.\n",
                     chain.actions[i].tool_name.c_str(), start_idx);

        for (int j = start_idx; j >= 0; --j) {
          if (is_action_allowed(chain.actions[j], caller_lib)) {
            ah_debug_log("ah_get_next_hop: Routing to tool '%s' trampoline\n",
                         chain.actions[j].tool_name.c_str());
            return chain.actions[j].trampoline_ptr;
          }
        }

        ah_debug_log("ah_get_next_hop: Filters exhausted. Routing to native OS "
                     "pointer (%p)\n",
                     chain.native_os_ptr);
        return chain.native_os_ptr;
      }
    }
  }

  ah_debug_log(
      "ah_get_next_hop: orig_out not found in chain! Returning NULL\n");
  return nullptr;
}

} // extern "C"

// -----------------------------------------------------------------------------
// Internal Interceptors
// -----------------------------------------------------------------------------

static void *(*real_dlsym)(void *, const char *) = nullptr;

extern "C" void *audit_dlsym_wrapper(void *handle, const char *symbol) {
  if (tls_hooks_paused || tls_hooks_ignored) {
    return real_dlsym ? real_dlsym(handle, symbol) : nullptr;
  }

  {
    std::shared_lock lock(*get_hooks_mutex());
    auto it = get_hooks()->find(symbol);
    if (it != get_hooks()->end()) {
      auto &chain = it->second;

      if (chain.native_os_ptr == nullptr) {
        tls_hooks_paused = true;
        chain.native_os_ptr = real_dlsym(handle, symbol);
        tls_hooks_paused = false;
      }

      if (!chain.actions.empty()) {
        if (chain.force_dispatcher_bind) {
          return chain.actions.back().dispatcher_ptr;
        } else {
          if (chain.actions.front().original_out_ptr) {
            *(chain.actions.front().original_out_ptr) = chain.native_os_ptr;
          }
          return chain.actions.back().trampoline_ptr;
        }
      }
    }
  }

  if (real_dlsym) {
    return real_dlsym(handle, symbol);
  }

  return nullptr;
}

// -----------------------------------------------------------------------------
// LD_AUDIT Linker Callbacks
// -----------------------------------------------------------------------------

extern "C" {

unsigned int la_version(unsigned int version) {
  if (version == 0)
    return 0;
  return LAV_CURRENT;
}

void la_preinit(uintptr_t *cookie) {
  if (getenv("AH_DEBUG")) {
    ah_debug_enabled = true;
    ah_debug_log("Diagnostics Enabled.\n");
  }

  const char *plugins_env = getenv("AH_PLUGINS");
  if (plugins_env) {
    char *env_copy = strdup(plugins_env);
    if (env_copy) {
      char *saveptr = nullptr;
      char *plugin_path = strtok_r(env_copy, ":", &saveptr);

      while (plugin_path != nullptr) {
        void *handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
          fprintf(stderr, "[AuditCore] FATAL: Failed to load plugin '%s'\n",
                  plugin_path);
          fprintf(stderr, "[AuditCore] Error: %s\n", dlerror());
        } else {
          // Check if the plugin exports the objopen callback
          auto cb = reinterpret_cast<ah_plugin_on_objopen_t>(
              dlsym(handle, "ah_plugin_on_objopen"));
          if (cb) {
            plugin_objopen_callbacks.push_back(cb);
            ah_debug_log("la_preinit: Discovered ah_plugin_on_objopen in "
                         "plugin '%s'\n",
                         plugin_path);
          }
        }
        plugin_path = strtok_r(nullptr, ":", &saveptr);
      }
      free(env_copy);
    }
  }
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
  const char *libname = nullptr;
  if (map) {
    std::unique_lock lock(*get_objects_mutex());
    if (map->l_name && map->l_name[0] != '\0') {
      (*get_objects())[*cookie] = map->l_name;
      libname = map->l_name;
    } else {
      (*get_objects())[*cookie] = program_invocation_short_name;
      libname = program_invocation_short_name;
    }
  }

  // Broadcast the event to all registered plugins
  if (libname) {
    for (auto cb : plugin_objopen_callbacks) {
      cb(libname, *cookie);
    }
  }

  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

unsigned int la_objclose(uintptr_t *cookie) {
  std::unique_lock lock(*get_objects_mutex());
  get_objects()->erase(*cookie);
  return 0;
}

static uintptr_t process_symbind(const char *symname, uintptr_t original_addr,
                                 unsigned int *flags, uintptr_t *refcook) {
  if (strcmp(symname, "dlsym") == 0) {
    real_dlsym =
        reinterpret_cast<void *(*)(void *, const char *)>(original_addr);
    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
    return reinterpret_cast<uintptr_t>(&audit_dlsym_wrapper);
  }

  if (strcmp(symname, "ah_set_caller_filter") == 0) {
    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
    using filter_ptr_t = int (*)(const char *, ah_filter_mode_t, const char **,
                                 size_t);
    return reinterpret_cast<uintptr_t>(
        static_cast<filter_ptr_t>(&ah_set_caller_filter));
  }

  if (strcmp(symname, "ah_set_target") == 0) {
    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
    using set_target_ptr_t = int (*)(const char *, void *);
    return reinterpret_cast<uintptr_t>(
        static_cast<set_target_ptr_t>(&ah_set_target));
  }

  if (strcmp(symname, "ah_get_target") == 0) {
    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
    using get_target_ptr_t = ah_err_t (*)(const char *, void **);
    return reinterpret_cast<uintptr_t>(
        static_cast<get_target_ptr_t>(&ah_get_target));
  }

  if (tls_hooks_paused || tls_hooks_ignored) {
    return original_addr;
  }

  std::shared_lock lock(*get_hooks_mutex());
  auto it = get_hooks()->find(symname);
  if (it != get_hooks()->end()) {
    auto &chain = it->second;

    if (chain.native_os_ptr == nullptr) {
      chain.native_os_ptr = reinterpret_cast<void *>(original_addr);
    }

    if (chain.force_dispatcher_bind) {
      ah_debug_log("process_symbind: Binding '%s' to Dynamic Dispatcher due "
                   "to force_dynamic request\n",
                   symname);
      *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
      return reinterpret_cast<uintptr_t>(chain.actions.back().dispatcher_ptr);
    }

    std::string caller_lib;
    {
      std::shared_lock obj_lock(*get_objects_mutex());
      auto obj_it = get_objects()->find(*refcook);
      if (obj_it != get_objects()->end()) {
        caller_lib = obj_it->second;
      }
    }

    for (int i = chain.actions.size() - 1; i >= 0; --i) {
      if (is_action_allowed(chain.actions[i], caller_lib)) {
        if (chain.actions.front().original_out_ptr) {
          *(chain.actions.front().original_out_ptr) = chain.native_os_ptr;
        }
        ah_debug_log("process_symbind: Binding '%s' to Static Trampoline '%s'\n",
                     symname, chain.actions[i].tool_name.c_str());
        *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
        return reinterpret_cast<uintptr_t>(chain.actions[i].trampoline_ptr);
      }
    }
  }

  return original_addr;
}

uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
                       uintptr_t *defcook, unsigned int *flags,
                       const char *symname) {
  return process_symbind(symname, sym->st_value, flags, refcook);
}

uintptr_t la_symbind32(Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
                       uintptr_t *defcook, unsigned int *flags,
                       const char *symname) {
  return process_symbind(symname, sym->st_value, flags, refcook);
}

} // extern "C"