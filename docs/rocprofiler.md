

# API Wrappers
- When writing wrapper functions, if relying on other components, take a reference to the 
  component to make sure the dependency is initialized before the current component. You can then
  assume the ordering is preserved by rocprofiler-sdk.