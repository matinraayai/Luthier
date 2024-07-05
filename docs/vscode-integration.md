

```json
{
    "configurations": [
        {
            "includePath": [
                "/opt/rocm/include",
                "/opt/rocm/llvm/include",
                "${workspaceRoot}/include"
            ],
            "defines": [
                "AMD_INTERNAL_BUILD",
                "LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"
            ],
            "cppStandard": "c++20",
            "name": "Linux",
            "compileCommands": "${workspaceRoot}/.vscode/compile_commands.json",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```