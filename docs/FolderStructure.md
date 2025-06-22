# High-level Folder Structure of Luthier
The Luthier project consists of the following folders:
- [`.devcontainer/`](../.devcontainer/): Contains the [`devcontainer.json`](../.devcontainer/devcontainer.json) file.
It can be used with IDEs like CLion or VSCode to run and develop Luthier inside a container. By default, it uses
the dev container provided by us, but you can build the container locally inside the [`dockerfiles`](../dockerfiles) folder.
- [`.vscode`](../.vscode): contains example cmake configuration files which can be used with VSCode to enable 
Intellisense code completion.
- [`cmake`](../cmake): contains CMake modules and CMake installation configuration files.
- [`dockerfiles`](../dockerfiles): contains the Dockerfiles of the Luthier project to run and develop Luthier-based
tools.
- [`docs`](../docs): contains the Luthier documentation.
- [`examples`](../examples): contains example Luthier tools.
- [`include`](../include): contains the public-facing API of Luthier.
- [`src`](../src): contains the Luthier source code and tests.