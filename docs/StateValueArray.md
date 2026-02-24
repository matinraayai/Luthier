# State Value Array (SVA)

## Overview
The **State Value Array (SVA)** is a specialized data structure consisting of **64 32-bit (DWORD) values**. In the context of a GPU wavefront, these 64 values are mapped across the lanes of a single **VGPR** or **AGPR**.



## Why We Need It
The SVA is in charge of:

* **Shared Wavefront Storage:** Providing a consistent location for values shared among all lanes in a wavefront.
* **Register Spilling for Branch Relaxation:** Since our instrumentation does not have access to the original application's stack, we lack a safe place to spill registers.
* **Preventing Instruction Overflow:** Hooks can easily push an `S_BRANCH` target beyond its 16-bit relative offset limit . The SVA allows us to spill SGPRs, enabling **branch relaxation**—the process of converting short branches into 32-bit long jumps without clobbering application data.

## How We Determine Which Register to Use
The placement of the SVA is determined by the `LRStateValueStorageAndLoadLocationsAnalysis` pass:

### 1. Target-Specific Allocation
| Hardware Generation | SVA Storage Options | Notes |
| :--- | :--- | :--- |
| **Pre-GFX908** | VGPR only | AGPRs do not exist on these targets. |
| **Post-GFX908** | VGPR or AGPR | AGPRs can be used in vector instructions, allowing us to save VGPRs for application logic. |

### 2. Liveness and Register Scavenging
If no unused registers are available throughout the entire program, we use **Liveness Analysis** to identify registers that are dead at specific points in the execution.
* The SVA is moved into these dead registers dynamically.
* Because the SVA's location can change, its current location is stored in **PC Sections** so the code knows where to find its data.



### 3. Segmented Load Plans
To manage the SVA across complex control flow, we utilize a customized **Slot Indexes analysis**. This is specifically modified for **Predicated Machine Basic Blocks** (blocks that terminate or change behavior when the `EXEC` mask is modified).
* **Slot Indexes:** Every instruction and basic block start is assigned a unique index.
* **Segmented Logic:** This allows the compiler to change the Load Plan (where the SVA is stored and how it’s accessed) in discrete segments of the code.
