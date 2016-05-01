#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>

// Include the stuff for OpenCL
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// Include the stuff for R
#include <Rcpp.h>
using namespace Rcpp;

// Store the kernel source code in an array of lines
const int SOURCE_LINES = 4;
const char* source[SOURCE_LINES] = {
  "kernel void batchEP(global float* y, global float* x, global float* beta) {",
  "  size_t i = get_global_id(0);",
  "  beta[i] = y[i] + x[i];",
  "}"
};

void ep_sequential(float* y, float* x, int n, int m, int k, float* beta) {
  stop("sequential ep not yet implemented.");
} // end ep_sequential

void ep_parallel(float* y, float* x, int n, int m, int k, float* beta) {
  char name[128];
  cl_int err;
  cl_platform_id platform;
  cl_uint deviceCount;
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  
  // Get the platform
  clGetPlatformIDs(1, &platform, NULL);
  
  // Get the device count
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
  
  // Get all of the devices
  cl_device_id devices[deviceCount];
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
  
  // Choose the device with the most compute units
  int max_cus = 0;
  for (int i = 0; i < deviceCount; i++) {
    int this_cus = 0;
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &this_cus, NULL);
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, NULL);
    fprintf(stdout, "Device %d: %s (%d CUs)\n", i, name, this_cus);
    
    if (this_cus > max_cus){
      device = devices[i];
      max_cus = this_cus;
    } // end if
  } // end for
  
  // Print out the device name
  clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
  fprintf(stdout, "Using: %s\n", name);
  
  // Create the context for the device
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    stop("error");

  // Create the program from the source code 
  program = clCreateProgramWithSource(context, SOURCE_LINES, source, NULL, NULL);
  if (err != CL_SUCCESS)
    stop("program could not be created from program source");
  
  // Build the program
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  // Create the kernel
  kernel = clCreateKernel(program, "batchEP", &err);
  if (err != CL_SUCCESS)
    stop("kernel could not be created");

  
  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
    stop("command queue could not be created");

  
  // Set the input memory
  cl_mem mem_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n * k,  x, &err);
  cl_mem mem_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n * m,  y, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate input buffer");
  
  // Set the output memory
  cl_mem mem_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, NULL, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate output buffer");
  
  // Set the parameters
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in1);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_in2);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_out);
  
  // Set the kernel for execution
  int dim = 1;
  size_t dims[3] = {n, 0, 0};
  clEnqueueNDRangeKernel(queue, kernel, dim, NULL, dims, NULL, 0, NULL, NULL);
  
  // Execute
  clFlush(queue);
  clFinish(queue);
  
  // Read out our results
  if (clEnqueueReadBuffer(queue, mem_out, CL_TRUE, 0, sizeof(float) * n, beta, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read output");
  
  // Clean up OpenCL resources
  clReleaseMemObject(mem_in1);
  clReleaseMemObject(mem_in2);
  clReleaseMemObject(mem_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
} // end ep_parallel

// [[Rcpp::export]]
List survivalEP(const NumericMatrix y, const NumericMatrix x, // input
                const int max_iter, const float thresh,       // stoping criteria
                bool async) {
  // Get the dimension of the matrices
  const int y_cols = y.cols();
  const int y_rows = y.rows();
  const int x_cols = x.cols();
  const int x_rows = x.rows();
  
  // Check if the vectors are the same size
  if (y_rows != x_rows)
    stop("matrices not the same length");
  
  // Initialize outputs
  NumericMatrix beta(y_rows, 1);
  
  // Convert the data to float arrays
  float *y_fl = new float[y_rows * y_cols];
  float *x_fl = new float[x_rows * x_cols];
  float *beta_fl = new float[y_rows];
  for (int i = 0; i < y_rows * y_cols; i++)
    y_fl[i] = (float)y[i];
  for (int i = 0; i < x_rows * x_cols; i++)
    x_fl[i] = (float)x[i];
  for (int i = 0; i < y_rows; i++)
    beta_fl[i] = 0;
  
  // Iterations
  for (int i = 0; i < max_iter; i++) {
    // implement algorithm
    if (async)
      ep_parallel(y_fl, x_fl, y_rows, y_cols, x_cols, beta_fl);
    else
      ep_sequential(y_fl, x_fl, y_rows, y_cols, x_cols, beta_fl);
  } // end for
  
  // Extract results
  for (int i = 0; i < y_rows; i++)
    beta[i] = beta_fl[i];
  
  // Release memory
  delete [] y_fl;
  delete [] x_fl;
  delete [] beta_fl;
  
  // Return list
  List out;
  out["y"] = y;
  out["x"] = x;
  out["beta"] = beta;
  
  return out;
} // end survivalEP