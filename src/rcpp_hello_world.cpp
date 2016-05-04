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
const int VEC_SOURCE_LINES = 4;
const char* vec_source[VEC_SOURCE_LINES] = {
  "kernel void vectorAdd(global float* input1, global float* input2, global float* output) {",
  "  size_t i = get_global_id(0);",
  "  output[i] = input1[i] + input2[i];",
  "}"
};

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];
    int cus, freq;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &cus, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(int), &freq, NULL);
    
    fprintf(stdout, "%s : %s (%i CUs @ %i MHz)\n", vendor, name, cus, freq);
} // end print_device_info

// A utility function that checks that our kernel execution performs the
// requested work over the entire range of data.
static int validate(float* input1, float* input2, float* output, int vecLen) {
  int i;
  bool bad = false;
  for (i = 0; i < vecLen; i++) {
    fprintf(stdout, "%d: %f + %f -> %f\n", i, input1[i], input2[i], output[i]);
    
    // Check the output
    if (output[i] != (input1[i] + input2[i]))
      bad = true;
  } // end for
  
  if (bad) 
    stop("did not validate");

  return 0;
} // end validate

NumericVector vectorAdd(NumericVector a, NumericVector b) {
  char name[128];
  int vecLen = 1024;
  cl_int err;
  cl_platform_id platform;
  cl_uint deviceCount;
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  
  // Check if the vectors are the same size
  if (a.length() != b.length())
    stop("vectors not the same length");
  
  // Get the length of our vector
  vecLen = a.length();
  
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
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Create the program from the source code 
  program = clCreateProgramWithSource(context, VEC_SOURCE_LINES, vec_source, NULL, NULL);
  if (err != CL_SUCCESS) {
    stop("program could not be created from program source");
  } // end if
  
  // Build the program
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  // Create the kernel
  kernel = clCreateKernel(program, "vectorAdd", &err);
  if (err != CL_SUCCESS) {
    stop("kernel could not be created");
  } // end if
  
  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    stop("command queue could not be created");
  } // end if
  
  // test data
  
  // Create the test data
  float* test_in1 = (float*)malloc(sizeof(float) * vecLen);
  float* test_in2 = (float*)malloc(sizeof(float) * vecLen);
  float* test_out = (float*)malloc(sizeof(float) * vecLen);
  for (int i = 0; i < vecLen; i++) {
    test_in1[i] = (float)a[i];
    test_in2[i] = (float)b[i];
  } // end for
  
  // end test data
  
  // Set the input memory
  cl_mem mem_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecLen,  test_in1, &err);
  cl_mem mem_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecLen,  test_in2, &err);
  if (err != CL_SUCCESS) {
    stop("failed to allocate input buffer");
  } // end if
  
  // Set the output memory
  cl_mem mem_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * vecLen, NULL, &err);
  if (err != CL_SUCCESS) {
    stop("failed to allocate output buffer");
  } // end if
  
  // Set the parameters
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in1);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_in2);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_out);
  
  // Set the kernel for execution
  int dim = 1;
  size_t dims[3] = {vecLen, 0, 0};
  clEnqueueNDRangeKernel(queue, kernel, dim, NULL, dims, NULL, 0, NULL, NULL);
  
  // Execute
  clFlush(queue);
  clFinish(queue);
  
  // Read out our results
  if (clEnqueueReadBuffer(queue, mem_out, CL_TRUE, 0, sizeof(float) * vecLen, test_out, 0, NULL, NULL) != CL_SUCCESS) {
    stop("failed to read output");
  }
  
  // Check to see if the kernel did what it was supposed to:
  if (validate(test_in1, test_in2, test_out, vecLen)) {
    fprintf(stdout, "All values were properly added.\n");
  } // end if
  
  // Clean up OpenCL resources
  clReleaseMemObject(mem_in1);
  clReleaseMemObject(mem_in2);
  clReleaseMemObject(mem_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  
  // And sytem resources
  free(test_in1);
  free(test_in2);
  free(test_out);
  
  // Convert array to R vector
  NumericVector c(vecLen);
  for (int i = 0; i < vecLen; i++)
    c[i] = test_out[i];
  
  return c;
} // end vectorAdd
