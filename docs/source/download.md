# Download

## v{RELEASE} release

* Download [TGZ](https://github.com/openucx/ucx/releases/download/v{RELEASE}/ucx-{RELEASE}.tar.gz) [SRPM](https://github.com/openucx/ucx/releases/download/v{RELEASE}/ucx-{RELEASE}-1.fc30.src.rpm)
* [Release page](https://github.com/openucx/ucx/releases/tag/v{RELEASE})
* [Running](running)

#### Features
- Added support for multiple listening transports
- Added UCT socket-based connection manager transport
- Updated API for UCT component management
- Added API to retrieve the listening port
- Added UCP active message API
- Removed deprecated API for querying UCT memory domains
- Refactored server/client examples
- Added support for dlopen interception in UCM
- Added support for PCIe atomics
- Updated Java API: added support for most of UCP layer operations
- Updated support for Mellanox DevX API
- Added multiple UCT/TCP transport performance optimizations
- Optimized memcpy() for Intel platforms
- Added protection from non-UCX socket based app connections
- Improved search time for PKEY object
- Enabled gtest over IPv6 interfaces
- Updated Mellanox and Bull device IDs
- Added support for CUDA_VISIBLE_DEVICES
- Increased limits for CUDA IPC registration

#### Bugfixes
- Multiple fixes in UCP, UCT, UCM libraries
- Multiple fixes for BSD and Mac OS systems
- Fixes for Clang compiler
- Fix CPU optimization configuration options
- Fix JUCX build on GPU nodes
- Fix in Azure release pipeline flow
- Fix in CUDA memory hooks management
- Fix in GPU memory peer direct gtest
- Fix in TCP connection establishment flow
- Fix in GPU IPC check
- Fix in CUDA Jenkins test flow
- Multiple fixes in CUDA IPC flow
- Fix adding missing header files
- Fix to prevent failures in presence of VPN enabled Ethernet interfaces

<br/>

## Previous releases

[GitHub release page](https://github.com/openucx/ucx/releases)
