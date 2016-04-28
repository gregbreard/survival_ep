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

// Hard-coded number of values to test, for convenience.
#define NUM_VALUES 1024

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
static int validate(int* input1, int* input2, int* output) {
    int i;
    for (i = 0; i < NUM_VALUES; i++) {
        
        // The kernel was supposed to add the vectors.
        if ( output[i] != (input1[i] + input2[i]) ) {
            fprintf(stdout,
                    "Error: Element %d did not match expected output.\n", i);
            fprintf(stdout,
                    "       Saw %d, expected %d\n", output[i], input1[i] + input2[i]);
            fflush(stdout);
            return 0;
        } // end if
    } // end for
    
    return 1;
} // end validate


// [[Rcpp::export]]
List rcpp_hello_world() {

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}

// [[Rcpp::export]]
List vectorAdd(NumericVector a, NumericVector b) {
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
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Read in the kernel source
  std::ifstream source("ep_kernel.cl");
  std::string sourceCode(std::istreambuf_iterator<char>(source), (std::istreambuf_iterator<char>()));
  fprintf(stdout, "Kernel Source:\n %s\n", sourceCode.c_str());
  
  // Just to do something in R  
  List c = List::create(a, b);
  return c;
}

// [[Rcpp::export]]
List vectorAdd2(NumericVector a, NumericVector b) {
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
    if (this_cus > max_cus){
      device = devices[i];
      max_cus = this_cus;
    } // end if
  } // end for
    
  // Print out the device name
  //clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
  //fprintf(stdout, "Using: %d\n", device);
  //fprintf(stdout, "test %d\n", 1);
  
  // Create the context for the device
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
    
  // Read in the kernel source
  std::ifstream sourceFile("ep_kernel.cl");
  std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
   
  // Create the program from the source code 
  // (we need to wrap the string in an array list for the function)
  const char **cptr;
  cptr = (const char **) malloc(sizeof(char*));
  cptr[1] = sourceCode.c_str();
  program = clCreateProgramWithSource(context, 1, cptr, NULL, &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Build the program
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  // Create the kernel
  kernel = clCreateKernel(program, "vectorAdd", &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // test
  
  // Create the test data
  int* test_in1 = (int*)malloc(sizeof(int) * NUM_VALUES);
  int* test_in2 = (int*)malloc(sizeof(int) * NUM_VALUES);
  int* test_out = (int*)malloc(sizeof(int) * NUM_VALUES);
  for (int i = 0; i < NUM_VALUES; i++) {
    test_in1[i] = (int)i;
    test_in2[i] = (int)(NUM_VALUES - i);
  } // end for
  
  // /test
  
  // Set the input memory
  cl_mem mem_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_VALUES,  test_in1, &err);
  cl_mem mem_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_VALUES,  test_in2, &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Set the output memory
  cl_mem mem_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * NUM_VALUES, NULL, &err);
  if (err != CL_SUCCESS) {
    stop("error");
  } // end if
  
  // Set the parameters
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in1);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_in2);
  
  // Set the kernel for execution
  int dim = 1;
  size_t dims[3] = {NUM_VALUES, 0, 0};
  clEnqueueNDRangeKernel(queue, kernel, dim, NULL, dims, NULL, 0, NULL, NULL);
  
  // Execute
  clFlush(queue);
  clFinish(queue);
    
  // Read out our results
  clEnqueueReadBuffer(queue, mem_out, CL_TRUE, 0, sizeof(int) * NUM_VALUES, test_out, 0, NULL, NULL);
  
  // Check to see if the kernel did what it was supposed to:
  if (validate(test_in1, test_in2, test_out)) {
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
  
  // Just to do something in R  
  List c = List::create(a, b);
  return c;
}