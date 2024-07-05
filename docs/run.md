# Running Luthier-based Tools
To instrumentAndLoad HIP-based applications using tools developed with Luthier, run the following command:
```shell
LD_PRELOAD=${LUTHIER_PATH}:${TOOL_PATH} ${APPLICATION_CMD}
```
- ```LD_PRELOAD``` loads the specified shared objects before loading any other library
- ```${LUTHIER_PATH}``` is the path to Luthier's shared library
- ```${TOOL_PATH}``` is the path to the tool built as a shared object using Luthier
- ```${APPLICATION_CMD}``` is the command one uses to launch the target application

If the target application is not HIP-based, then the environment variable ```HIP_ENABLE_DEFERRED_LOADING``` must be
set to 0 to ensure the HIP runtime loads the instrumentation device functions.