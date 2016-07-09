#include <cuda_runtime.h>
#include <iostream>
#include <exception>


void cuda_error(cudaError_t e, int code_line) {
    if(e != cudaSuccess) {
        std::cerr << "CUDA execution error: " << e << " at line " << code_line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(e) (cuda_error(e, __LINE__))


// Function to gather information about devices prior to memory allocation
void print_cuda_device_properties() {
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for(int i = 0; i < count; ++i) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "===== Graphics Device Information: GPU" << i << " =====    " << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Clock rate: " << prop.clockRate << std::endl;
        std::cout << "Device copy overlap: ";
        if(prop.deviceOverlap)
            std::cout << "Enabled" << std::endl;
        else
            std::cout << "Disabled" << std::endl;
        std::cout << "Kernel execition timeout: ";
        if(prop.kernelExecTimeoutEnabled)
            std::cout << "Enabled" << std::endl << std::endl;
        else
            std::cout << "Disabled" << std::endl << std::endl;

        std::cout << "===== Graphics Device Memory Information: GPU #" << i << " =====    " << std::endl;
        std::cout << "Total global memory: " << (float)prop.totalGlobalMem / 1000000000 << " GB" << std::endl;
        std::cout << "Total constant memory: " << (float)prop.totalConstMem / 1000 << " KB" << std::endl;
        std::cout << "Max memory pitch: " << (float)prop.memPitch / 1000000000 << " GB" << std::endl;
        std::cout << "Texture alignment: " << prop.textureAlignment << std::endl << std::endl;

        std::cout << "===== Graphics Device MultiProcessor Information: GPU #" << i << " =====    " << std::endl;
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared memory per MP: " << (float)prop.sharedMemPerBlock / 1000 << " KB" << std::endl;
        std::cout << "Registers per MP: " << prop.regsPerBlock << std::endl;
        std::cout << "Threads in warp: " << prop.warpSize << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max thread dimensions: " << prop.maxThreadsDim[0] << ' ' << prop.maxThreadsDim[1] << ' ';
        std::cout << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid dimensions: " << prop.maxGridSize[0] << ' ' << prop.maxGridSize[1] << ' ';
        std::cout << prop.maxGridSize[2] << std::endl << std::endl << std::endl << std::endl;
    }

}

int main() {
    print_cuda_device_properties();
    return 0;
}

