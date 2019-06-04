#define CL_HPP_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
//#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#include <CL/cl.h>
#include "cl.hpp"

#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   std::ifstream fin("input.txt");
   std::ofstream fout("output.txt");
   size_t N, M;
   fin >> N >> M;

    size_t const block_size = 16;

    int mat1_size = N;
    while (mat1_size % block_size) {
        mat1_size++;
    }

    std::cerr << mat1_size << "\n";

    int mat2_size = M;
    while (mat2_size % block_size) {
        mat2_size++;
    }

   std::vector<double> a(N * N);
   std::vector<double> b(M * M);
   std::vector<double> c(N * N);

   for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
         fin >> a[i * N + j];
      }
   }

   for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < M; j++) {
         fin >> b[i * M + j];
      }
   }

   try {
      // create platform
      cl::Platform::get(&platforms);

      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);


      // load opencl source
      std::ifstream cl_file("matrix_convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));


      // create program
      cl::Program program(context, source);

      // compile opencl source
      try
      {
         program.build(devices);
      }
      catch (cl::Error const & e)
      {         
         std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
         std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
         std::cout << log_str;
         return 0;
      }

      // create a message to send to kernel

      size_t const base_matrix_size = N * N;
      size_t const kernel_matrix_size = M * M;

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * base_matrix_size);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * kernel_matrix_size);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * base_matrix_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * base_matrix_size, &a[0]);
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * kernel_matrix_size, &b[0]);

      // load named kernel from opencl source
      cl::Kernel kernel_gmem(program, "convolve");

      cl::KernelFunctor convolve(kernel_gmem, queue, cl::NullRange, cl::NDRange(mat1_size, mat1_size), cl::NDRange(block_size, block_size));

      cl::Event event = convolve(dev_a, dev_b, dev_c, static_cast<int>(M), static_cast<int>(N));

      event.wait();
      cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      std::cerr << "Total time: " << (end - start) / 1e6 << std::endl;

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * base_matrix_size, &c[0]);

      for (size_t i = 0; i < N; ++i) {
         for (size_t j = 0; j < N; ++j) {
            fout << c[i * N + j] << " ";
         }
         fout << std::endl;
      }
   }
   catch (cl::Error const & e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
