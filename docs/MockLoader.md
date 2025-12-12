# The AMD GPU Mock Loader

## Introduction

The mock loader is a very useful tool for Luthier developers that streamlines the instrumentation pass development
process, allowing them to easily experiment with different possible instrumentation scenarios without the constraints
of physical access to an AMD GPU device to load their target code. In this section we go over Luthier's
`MockAMDGPULoader` design and its usage.

## Background

By design, Luthier's lifting and inspection passes only inspect the contents of executable memory regions on a
target device, and use code objects as an optional secondary source of information or "hint" if they are available.
Exclusively running the inspection passes directly on GPU memory has the following issues:

1. It makes testing the inspection passes cumbersome and hard to scale, as having physical access to
   all supported devices is almost always not feasible.
2. It provides no way to instrument code objects offline without first loading them onto a target device.

## Design

A new loader implementation called the "Mock AMDGPU Loader" is introduced to dynamically load and link AMD GPU code
objects on host memory instead of device memory, thereby "mocking" or emulating the normal loading process and
eliminating the need for a physical AMD GPU to utilize Luthier's instrumentation passes.

The mock loader is a stripped down version of the ROCr HSA loader and behaves somewhat similarly to the code loading
APIs provided by the HSA standard. After creating a loader instance, the users can request a `MockAMDGPUExecutable`. The
mock executable can then be used to load code objects

These two steps will allow the instrumentation passes to seamlessly

`MockAMDGPULoader` mechanism is introduced. The mock loader dynamically loads and links
AMD GPU code objects on the host memory instead of the target device's memory, thereby "mocking" the normal loading
process. he mock loader developers can test their instrumentation passes for all intended AMD GPU targets without having
physical access to the GPUs on their system.

A mechanism that "mocks"
the code object loading and linking process of an AMD GPU runtime is desired, which loads the code objects on host
memory instead of a target AMD GPU device.

## Design

The mock loader's implementation is a very stripped down version of the HSA loader.

1. Emulate the
2.
3. primarily allow for easier and faster
   testing and experimentation with the loaded code objects and the instrumentation passes.

To allow for easier
testing and development of instrumentation passes on all supported AMD GPU target devices while eliminating the need of
having a physical AMD GPU attached to the test system,

1. Testing and

To allow for easier testing and experimentation,
Because of this,

1. to allow Luthier developers to test and experiment with the
   instrumentation passes without requiring the presence of a physical AMD GPU.

This requirement makes testing and development of the instrumentation passes challenging, as
they have to rely as without
a
the other hand, However, it hinders testing as all inspection tests require the presence of a target AMD GPU runtime
and a target device, making

In addition to testing

To remedy
Originally, an "offline" mode for inspection passes was considered to
support instrumentation of code objects that has not been loaded into a target runtime. The passes then could which the
passes would behave differently if they are only dealing with

1.
2. It

To streamline the instrumentation pass development process, Luthier has

## Constraints on The Code Objects Used In A Mock Executable

- The mock loader is intentionally not thread-safe to reduce implementation complexity.
