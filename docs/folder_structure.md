# High-level Folder Structure of Luthier

The Luthier project consists of the following folders:

- [`.devcontainer/`](../.devcontainer): Contains the
- [`devcontainer.json`](../.devcontainer/devcontainer.json) file. It can be used
  with IDEs like CLion or VSCode to run and develop Luthier inside a container.
  By default, it uses the dev container built by us, but you can build the
  container yourself from the [`dockerfiles`](../dockerfiles) folder.
- [`Plugins`](../src/lib/Plugins): contains the LLVM compiler
  plugins for Luthier. It is currently used by Luthier tools; In the future, it
  will contain compiler plugins used in static instrumentation as well.
- [`dockerfiles`](../dockerfiles): contains the Dockerfiles of the Luthier
  project to run and develop Luthier-based
  tools.
- [`docs`](../docs): contains the Luthier documentation.
- [`examples`](../examples): contains example Luthier tools.
- [`include`](../include): contains the public-facing API of Luthier.
- [`src`](../src): contains the Luthier source code.
- [`tests`](../tests): contains integration tests.