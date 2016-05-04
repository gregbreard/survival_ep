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

//"float dnorm(float x) {\n",
//"  //float val = x;//val = (1 / sqrt(2 * M_PI)) * exp(-1 * pow(x, 2) / 2);\n",
//"  return x;\n"
//"}\n",

// Store the kernel source code in an array of lines
const int SOURCE_LINES = 19;
const char* source[SOURCE_LINES] = {
  "// work around for missing M_PI\n",
  "constant float PI = 3.14159265358979323846;\n",
  "float dnorm(float x) {\n",
  "  return (1 / sqrt(2 * PI)) * exp(-1 * (x * x) / 2);\n",
  "}\n",
  "float pnorm(float x, int n) {\n",
  "  float sum = 0;\n",
  "  float delta = (x + 5) / n;\n",
  "  float i = -5;\n",
  "  while (i <= x) {\n",
  "    sum += dnorm(i);\n",
  "    i += delta;\n",
  "  }\n",
  "  return sum * delta;\n",
  "}\n",
  "kernel void batchEM(global float* y, global float* x, global float* beta) {\n",
  "  size_t i = get_global_id(0);\n",
  "  beta[i] = pnorm(y[i], 1000);\n",
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
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &this_cus, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, NULL);
      Rcout << "Device " << i << ": " << name << " (" << this_cus << " CUs)" << std::endl;
      
      if (this_cus > max_cus){
        device = devices[i];
        max_cus = this_cus;
      } // end if
    } // end for
    
    // Print out the device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    Rcout << "Using: " << name << std::endl;
    
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

void em_sequential(const arma::mat y, const arma::mat x, const arma::mat mu,
                   arma::mat* beta, arma::mat* eystar) {
  for (int i = 0; i < x.n_rows; i++) {
    if (y(i, 0) == 1)
      (*eystar)(i, 0) = mu(i, 0) + f(mu(i, 0));
    if (y(i, 0) == 0)
      (*eystar)(i, 0) = mu(i, 0) - g(mu(i, 0));
  } // end for
} // end em_sequential

void em_parallel(const arma::mat y, const arma::mat x, const arma::mat mu,
                 arma::mat* beta, arma::mat* eystar) {
  int y_rows = y.n_rows;
  int y_cols = y.n_cols;
  int x_rows = x.n_rows;
  int x_cols = x.n_cols;
  
  // Create float arrays for the data
  float *y_fl = new float[y_rows * y_cols];
  float *x_fl = new float[x_rows * x_cols];
  float *beta_fl = new float[x_cols];
  float *eystar_fl = new float[x_rows];
  
  // Copy the data to arrays
  for (int i = 0; i < y_rows; i++)
    for (int j = 0; j < y_cols; j++)
      y_fl[(i * y_rows) + j] = (float)y(i,j);
  for (int i = 0; i < x_rows; i++)
    for (int j = 0; j < x_cols; j++)
      x_fl[(i * x_rows) + j] = (float)x(i,j);
  for (int i = 0; i < x_cols; i++)
    beta_fl[i] = (*beta)(i, 0);
  for (int i = 0; i < x_rows; i++)
    eystar_fl[i] = (*eystar)(i, 0);
  
  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
    stop("command queue could not be created");
    
  // Set the input memory
  cl_mem mem_y_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * y_rows * y_cols,  y_fl, &err);
  cl_mem mem_x_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_rows * x_cols,  x_fl, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate input buffer");
  
  // Set the output memory
  cl_mem mem_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * x.n_rows, NULL, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate output buffer");
  
  // Set the parameters
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_y_in);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_x_in);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_out);
  
  // Set the kernel for execution
  int dim = 1;
  size_t dims[3] = {x.n_rows, 0, 0};
  clEnqueueNDRangeKernel(queue, kernel, dim, NULL, dims, NULL, 0, NULL, NULL);
  
  // Execute
  clFlush(queue);
  clFinish(queue);
  
  // Read out our results
  if (clEnqueueReadBuffer(queue, mem_out, CL_TRUE, 0, sizeof(float) * x.n_rows, beta_fl, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read output");
  
  // Extract results
  for (int i = 0; i < x.n_cols; i++)
    beta[i] = beta_fl[i];
  //for (int i = 0; i < x.n_rows; i++)
  //  eystar[i] = eystar_fl[i];
  
  // Release memory
  delete [] y_fl;
  delete [] x_fl;
  delete [] beta_fl;
  delete [] eystar_fl;
  
  // Clean up OpenCL resources
  clReleaseMemObject(mem_y_in);
  clReleaseMemObject(mem_x_in);
  clReleaseMemObject(mem_out);
  clReleaseCommandQueue(queue);
} // end em_parallel

// [[Rcpp::export]]
List survivalEM(const arma::mat y, const arma::mat x, // input
                const int max_iter, bool async) {
  // Get the dimension of the matrices
  //const int y_cols = y.n_cols;
  //const int y_rows = y.n_rows;
  //const int x_cols = x.n_cols;
  //const int x_rows = x.n_rows;
  
  // Check if the vectors are the same size
  if (y.n_rows != x.n_rows)
    stop("matrices not the same length");
  
  // Initialize outputs
  arma::mat beta(x.n_cols, 1);
  arma::mat eystar(x.n_rows, 1);
  beta.fill(0.0); 
  eystar.fill(0.0); 

  if (async) {
    // Load the OpenCL stuff here so we 
    // don't have to do it every iteration
    load_kernel();
  } // end if

  // Iterations
  for (int i = 0; i < max_iter; i++) {
    arma::mat mu = x * beta;
    
    // implement algorithm
    if (async)
      em_parallel(y, x, mu, &beta, &eystar);
    //else
      em_sequential(y, x, mu, &beta, &eystar);
      
    // maximization step
    beta = (x.t() * x).i() * x.t() * eystar;
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