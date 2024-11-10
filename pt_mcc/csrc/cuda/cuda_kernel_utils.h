/////////////////////////////////////////////////////////////////////////////
/// \file cuda_kernel_utils.h
///
/// \brief Utilities for the cuda implementations of the tensor operations.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,
///            Germany. See the LICENSE file at the top-level directory of
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KERNEL_UTILS_H_
#define CUDA_KERNEL_UTILS_H_

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline dim3 computeBlockGrid(const unsigned long long int pNumElements, const int pNumThreads)
{
    dim3 finalDimension(pNumElements / pNumThreads, 1, 1);
    finalDimension.x += (pNumElements % pNumThreads != 0) ? 1 : 0;

    // Limit the x dimension to 65535 and increment y if needed
    while (finalDimension.x > 65535)
    {
        finalDimension.y *= 2;
        finalDimension.x = (finalDimension.x + 1) / 2; // Ceil division

        // If y also exceeds 65535, move to z dimension
        if (finalDimension.y > 65535)
        {
            finalDimension.z *= 2;
            finalDimension.y = (finalDimension.y + 1) / 2; // Ceil division
        }
    }

    // Ensure none of the dimensions exceed 65535, return an error or adjust as needed
    if (finalDimension.x > 65535 || finalDimension.y > 65535 || finalDimension.z > 65535)
    {
        printf("Error: Block grid dimensions exceeded the limits!\n");
        return dim3(65535, 65535, 1); // Limit as a fallback, or handle error
    }

    return finalDimension;
}

#endif
