#ifndef CODE_OBJECT_MANAGER_HPP
#define CODE_OBJECT_MANAGER_HPP
#include <set>
#include <vector>

#include "hsa_code_object_reader.hpp"
#include "instrumentation_function.hpp"
#include "luthier_types.h"

namespace luthier::hsa {
class GpuAgent;

class ExecutableSymbol;
}// namespace luthier::hsa

namespace luthier {
/**
 * \brief A singleton object that keeps track of instrumentation functions and instrumented kernels
 * (per un-instrumented application kernel) in Luthier.
 */
class CodeObjectManager {
 public:
    CodeObjectManager(const CodeObjectManager &) = delete;
    CodeObjectManager &operator=(const CodeObjectManager &) = delete;

    static inline CodeObjectManager &instance() {
        static CodeObjectManager instance;
        return instance;
    }

    /**
     * Registers the wrapper kernel of tool's instrumentation functions
     * The arguments are captured by intercepting \p __hipRegisterFunction calls, and checking if \p __luthier_wrap
     * is in the name of the kernel
     * This function is called once per instrumentation function/kernel
     * \param wrapperHostPtr shadow host pointer of the instrumentation function's wrapper kernel
     * \param kernelName name of the wrapper kernel
     */
    void registerInstrumentationFunctionWrapper(const void *wrapperHostPtr, const char *kernelName);

    /**
     * Returns the \p hsa::ExecutableSymbol of the instrumentation function, given its HSA agent and wrapper kernel
     * host ptr
     * \param wrapperHostPtr shadow host pointer of the wrapper kernel
     * \param agent the GPU HSA agent where the instrumentation function is loaded on
     * \return the instrumentation function symbol
     */
    const hsa::ExecutableSymbol &getInstrumentationFunction(const void *wrapperHostPtr,
                                                            hsa::GpuAgent agent) const;

    /**
     * Returns the \p hsa::ExecutableSymbol of the wrapper kernel associated with an instrumentation function,
     * given its wrapper kernel shadow host pointer and the HSA GPU Agent it is loaded on
     * \param wrapperHostPtr shadow host pointer of the wrapper kernel
     * \param agent the HSA GPU Agent the wrapper kernel (and instrumentation function) is loaded on
     * \return the instrumentation function wrapper kernel's \p hsa::ExecutableSymbol
     */
    const hsa::ExecutableSymbol &getInstrumentationKernel(const void *wrapperHostPtr, hsa::GpuAgent agent) const;

    /**
     * Loads an instrumented \p hsa::Executable, containing the instrumented version of the \p originalKernel
     * Called by \p CodeGenerator after it has compiled an instrumentation ELF
     * \param instrumentedElf reference to the instrumented ELF file in memory
     * \param originalKernel the symbol of the target instrumented kernel
     */
    void loadInstrumentedKernel(const llvm::ArrayRef<uint8_t> &instrumentedElf,
                                const hsa::ExecutableSymbol &originalKernel);

    /**
     * Returns the instrumented kernel's \p hsa::ExecutableSymbol given its original un-instrumented version's KD
     * Used to run the instrumented version of the kernel when requested by the user
     * \param originalKernel symbol of the un-instrumented original kernel
     * \return symbol of the instrumented version of the target kernel
     * \throws std::runtime_error if the originalKernelKD is not found internally
     */
    const hsa::ExecutableSymbol &getInstrumentedKernel(const hsa::ExecutableSymbol &originalKernel) const;

 private:
    CodeObjectManager() = default;
    ~CodeObjectManager();

    /**
     * Iterates over all the frozen HSA executables in the HSA Runtime and registers the ones that belong to the
     * Luthier tool
     */
    void registerLuthierHsaExecutables() const;

    void processFunctions() const;

    /**
     * A set of all \p hsa::Executable handles that belong to the Luthier tool, containing the instrumentation function
     * and their wrapper kernels
     */
    // TODO: Replace these containers with LLVM-based containers
    mutable std::set<hsa::Executable> toolExecutables_{};

    /**
     * \brief A list of device functions captured by \p __hipRegisterFunction, not yet processed by
     * \p CodeObjectManager
     * The first \p std::tuple element is the "host shadow pointer" of the instrumentation function wrapper kernel,
     * created via the macro \p LUTHIER_EXPORT_FUNC
     * The second \p std::tuple element is the name of the dummy kernel
     */
    mutable std::vector<std::tuple<const void *, const char *>> unprocessedFunctions_{};

    mutable std::unordered_map<const void *, std::unordered_map<hsa::GpuAgent, luthier::InstrumentationFunction>>
        functions_{};

    std::unordered_map<hsa::ExecutableSymbol,
                               std::tuple<hsa::ExecutableSymbol, hsa::Executable, hsa::CodeObjectReader>>
        instrumentedKernels_{};
};
};// namespace luthier

#endif
