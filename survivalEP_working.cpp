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
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

const bool debug = false;

// Store the kernel source code in an array of lines
const int SOURCE_LINES = 20;
const char* source[SOURCE_LINES] = {
  "#define M_SQRT1_2PI_F 0.3989422804\n",
  "float dnorm(float x) {\n",
  "  return M_SQRT1_2PI_F * exp(-1 * (x * x) / 2);\n",
  "}\n",
  "float pnorm(float x) {\n",
  "  return ((1 + erf(x / M_SQRT2_F)) / 2);\n",
  "}\n",
  "float f(float mu) {\n",
  "  return (dnorm(-mu) / (1 - pnorm(-mu)));\n",
  "}\n",
  "float g(float mu) {\n",
  "  return (dnorm(-mu) / pnorm(-mu));\n",
  "}\n",
  "kernel void batchEM(global float* y, global float* mu, global float* eystar) {\n",
  "  size_t i = get_global_id(0);\n",
  "  if (y[i] == 1.0)\n",
  "    eystar[i] = mu[i] + f(mu[i]);\n",
  "  else if (y[i] == 0.0)\n",
  "    eystar[i] = mu[i] - g(mu[i]);\n",
  "}\n"
};

// Open CL objects
char name[128];
cl_int err;
cl_platform_id platform;
cl_uint deviceCount;
cl_device_id device;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
bool kernel_loaded = false;

// Loads the devices, etc for using Open CL
void load_kernel() {
  if (!kernel_loaded) {
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
      long mem = 0;
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &this_cus, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(long), &mem, NULL);
      if (debug) Rcout << "Device " << i << ": " << name << " (" << this_cus << " CUs) w " << mem << std::endl;
      
      if (this_cus > max_cus){
        device = devices[i];
        max_cus = this_cus;
      } // end if
    } // end for
    
    // Print out the device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    if (debug) Rcout << "Using: " << name << std::endl;
    
    // Create the context for the device
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
      stop("error");
    
    // Create the program from the source code 
    program = clCreateProgramWithSource(context, SOURCE_LINES, source, NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("program could not be created from program source");
    }
    
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("program could not be built");
    }
    
    // Create the kernel
    kernel = clCreateKernel(program, "batchEM", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("kernel could not be created");
    }
    
    // Don't need to reload
    kernel_loaded = true;
  } // end if
} // end load_kernel

// Releases the devices, etc used by Open CL
void release_kernel() {
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  
  // Make sure to reload next time
  kernel_loaded = false;
} // end release_kernel

double f (double mu) {
  double val = ((R::dnorm(-mu, 0, 1, false)) /
                  (1 - R::pnorm(-mu, 0, 1, true, false))
  ) ;
  return(val) ;
}

double g (double mu) {
  double val = ((R::dnorm(-mu, 0, 1, false)) /
                  (R::pnorm(-mu, 0, 1, true, false))
  ) ;
  return(val) ;
}

arma::mat em_sequential(const arma::mat y, const arma::mat mu) {
  // Initialize output
  arma::mat eystar(y.n_rows, 1);
  
  for (int i = 0; i < y.n_rows; i++) {
    if (y(i, 0) == 1)
      eystar(i, 0) = mu(i, 0) + f(mu(i, 0));
    if (y(i, 0) == 0)
      eystar(i, 0) = mu(i, 0) - g(mu(i, 0));
  } // end for
  
  return eystar;
} // end em_sequential

arma::mat em_parallel(const arma::mat y, const arma::mat mu) {
  // Initialize output
  arma::mat eystar(y.n_rows, 1);
  
  // Get the dimensions
  const int y_rows = y.n_rows;
  
  // Create float arrays for the data
  float *y_fl = new float[y_rows];
  float *mu_fl = new float[y_rows];
  float *eystar_fl = new float[y_rows];
  
  // Copy the data to arrays
  for (int i = 0; i < y_rows; i++)
      y_fl[i] = (float)y(i, 0);
  for (int i = 0; i < y_rows; i++) {
    mu_fl[i] = (float)mu(i, 0);
  } // end for
  if (debug) warning("got here 1");
  
  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
    stop("command queue could not be created");
  if (debug) warning("got here 2");
  
  // Set the input memory
  cl_mem y_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * y_rows,  y_fl, &err);
  cl_mem mu_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * y_rows,  mu_fl, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate input buffer");
  if (debug) warning("got here 3");
  
  // Set the output memory
  cl_mem eystar_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * y_rows, NULL, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate output buffer");
  if (debug) warning("got here 4");
  
  // Set the parameters
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &y_in);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mu_in);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &eystar_out);
  if (debug) warning("got here 5");
  
  // Set the kernel for execution
  int dim = 1;
  size_t dims[3] = {y_rows, 0, 0};
  clEnqueueNDRangeKernel(queue, kernel, dim, NULL, dims, NULL, 0, NULL, NULL);
  if (debug) warning("got here 6");
  
  // Execute
  clFlush(queue);
  clFinish(queue);
  if (debug) warning("got here 7");
  
  // Read out our results
  if (clEnqueueReadBuffer(queue, eystar_out, CL_TRUE, 0, sizeof(float) * y_rows, eystar_fl, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read output");
  if (debug) warning("got here 8");
  
  // Extract results
  for (int i = 0; i < y_rows; i++)
    eystar(i, 0) = eystar_fl[i];
  if (debug) warning("got here 9");
  
  // Release memory
  delete [] y_fl;
  delete [] mu_fl;
  delete [] eystar_fl;
  
  // Clean up OpenCL resources
  clReleaseMemObject(y_in);
  clReleaseMemObject(mu_in);
  clReleaseMemObject(eystar_out);
  clReleaseCommandQueue(queue);
  
  return eystar;
} // end em_parallel

// [[Rcpp::export]]
List survivalEM(const arma::mat y, const arma::mat x, // input
                const int max_iter, bool async) {
  // Check if the vectors are the same size
  if (y.n_rows != x.n_rows)
    stop("matrices not the same length");
  
  // Initialize outputs
  arma::mat beta(x.n_cols, 1);
  arma::mat eystar;
  beta.fill(0.0); 
  eystar.fill(0.0); 

  if (async) {
    // Load the OpenCL stuff here so we 
    // don't have to do it every iteration
    load_kernel();
  } // end if
  
  // Do some matrix stuff up front
  arma::mat x_t_x_i_x_t = (x.t() * x).i() * x.t();
  
  // Iterations
  for (int i = 0; i < max_iter; i++) {
    arma::mat mu = x * beta;
    
    // implement algorithm
    if (async)
      eystar = em_parallel(y, mu);
    else
      eystar = em_sequential(y, mu);
      
    // maximization step
    beta = x_t_x_i_x_t * eystar;
  } // end for
  
  if (async)
    release_kernel();
  
  // Return list
  List out;
  out["y"] = y;
  out["x"] = x;
  out["beta"] = beta;
  out["eystar"] = eystar;
  
  return out;
} // end survivalEM