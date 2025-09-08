## GPU Study

1. ROCm 이해하기 (https://rocm.docs.amd.com/en/latest/what-is-rocm.html). Conceptual 에서 GPU architecture documentation 다운 받아서 필요할 때 찾아서 보기.
2. hip 프로그래밍 가이드 (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html). 표시한 파트 읽어보기
3. 1-2 까지 됐다면 실제 구현체 뜯어보기. (Linux 의 경우 ROCr (ROCm runtime(ROCR) HSA runtime))
4. https://github.com/ROCm/rocm-systems 프로젝트 구조 살펴보기. 통합 이후 구조가 복잡해져서 어느 디렉토리에 어떤 내용이 있는지 알 필요가 있음.

```
projects/
  amdsmi/
  aqlprofile/
  clr/
  hip/
  hipother/
  hip-tests/
  rccl/
  rdc/
  rocm-core
  rocminfo/
  rocmsmilib/
  rocprofiler/
  rocprofiler-compute/
  rocprofiler-register/
  rocprofiler-sdk/
  rocprofiler-systems/
  rocrruntime/
  rocshmem/
  roctracer/
```

---

### AMD stack 공부

코드의 실행 흐름은 다음과 같습니다.

1. 상위 프레임워크(PyTorch 등)에서 ROCm API를 호출합니다.

2. ROCm 스택의 컴파일러가 이 요청에 필요한 GPU 코드를 생성합니다.

3. 생성된 GPU 코드는 ROCm 런타임을 거쳐 커널 드라이버에 전달됩니다.

4. 커널 드라이버는 이 코드를 받아 최종적으로 AMD GPU 하드웨어가 실행할 수 있도록 명령을 내립니다.


---

### ROCm Library

https://github.com/ROCm/rocm-libraries

```
projects/
  composablekernel/
  hipblas/
  hipblas-common/
  hipblaslt/
  hipcub/
  hipfft/
  hiprand/
  hipsolver/
  hipsparse/
  hipsparselt/
  miopen/
  rocblas/
  rocfft/
  rocprim/
  rocrand/
  rocsolver/
  rocsparse/
  rocthrust/
shared/
  rocroller/
  tensile/
  mxdatagenerator/
```

---
## ROCm systems

### init

**`clr/rocclr/device/rocm/rocdevice.cpp`**

```
init()
  hsa_init()
    Runtime::Acquire()
      runtime_singleton_->Load()
        Runtime::Load()
          thunkLoader_->Load()
            여기서 dlsym을 load한다.
          thunkLoader_->LoadThunkApiTable()
            여기에 hsakmt api들이 이어져있고, dlsym을 통해 가져온다.
          thunkLoader_->CreateThunkInstance
          AMD::Load()
            **amd_topology.cpp() → Load()**
              BuildTopology() ←**여기부터 다시 시작!!!**
                driver->GetNodeProperties()
                DiscoverCpu()
                  **runtime_singleton_->RegisterAgent()**
                DiscoverGpu()
                  **runtime_singleton_->RegisterAgent()**
                DiscoverAie()
                  **runtime_singleton_->RegisterAgent()**

```

BuildTopology()에서 driver에 해당하는 prop을 GetNodeProperties를 통해 얻는다(내부적으로 kmthsa/ioctl). 해당 정보를 통해 DiscoverXpu()에 주고, 이걸 기반으로 agent를 생성한다. agent를 register한다.


**`rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp`**

```
hsa_iterate_agents()
  Runtime::IterateAgent(**iterateAgentCallback**, nullptr)
    caller에서 받은 **iterateAgentCallback**을 실행한다.
  Device::iterateAgentCallback()
    hsa_agent_get_info()
      agent→GetInfo()
getDevices()
```

### g_devices
```
init()
  new Device()
  amd::Device::getDevices()

  device->Create()
  g_devices.push_back(device);

hipGetDeviceProperties()
  ihipGetDeviceProperties()
    g_devices에서 Device정보를 가져온다.
```
