#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <unordered_map>

class Sibir {
 public:
    Sibir() {};
    ~Sibir() {};
    static Sibir& getInstance();
    void *getOriginalHipSymbol(const std::string& sym) {return originalHipSymbols_[sym];}

    void intercept() {puts("intercepted");};

    static void init();
    static void destroy();
 private:
    static Sibir* sibir_;
    void recordAllOriginalHipSymbols();

    void recordOriginalHipSymbol(const char* sym);

    std::unordered_map<std::string, void*> originalHipSymbols_;
};
