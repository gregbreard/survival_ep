#include <cmath>
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

const bool DEBUG = false;

// The log2(rows) in the partial beta array to run the sequential sum on
const int CUTOFF = 8;

// Store the kernel source code in an array of lines
const int SOURCE_LINES = 50;
const char* source[SOURCE_LINES] = {
  "#define M_SQRT1_2PI_F 0.3989422804\n",
  "// probability functions\n",
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
  "// kernel for performing the expectation maximization \n",
  "kernel void expectation(global float* x, global float* y,\n",
  "                        global float* beta, global float* eystar,\n",
  "                        const int x_cols, const int x_rows) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  float mu = 0.0;\n",
  "  for (int l = 0; l < x_cols; l++)\n",
  "    mu += x[(row * x_cols) + l] * beta[l];\n",
  "  if (y[row] == 1.0)\n",
  "    eystar[row] = mu + f(mu);\n",
  "  else if (y[row] == 0.0)\n",
  "    eystar[row] = mu - g(mu);\n",
  "}\n"
  "// kernel for performing multiplication of each z(i,j) and y*(i) \n",
  "kernel void beta_part(global float* z, global float* eystar,\n",
  "                      global float* beta_part,\n",
  "                      const int x_rows) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  const size_t col = get_global_id(1);\n",
  "  beta_part[(col * x_rows) + row] = z[(col * x_rows) + row] * eystar[row];\n",
  "}\n"
  "// kernel for performing addition of each beta(i,j)\n",
  "kernel void part_sum(global float* beta_part, const int x_rows) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  const size_t col = get_global_id(1);\n",
  "  const size_t n = get_global_size(0);\n",
  "  const size_t add_row = row + n;\n",
  "  if (add_row < x_rows)\n",
  "    beta_part[(col * x_rows) + row] = beta_part[(col * x_rows) + row] + beta_part[(col * x_rows) + add_row];\n",
  "}\n"
  "// kernel for performing addition of each beta(i,j)\n",
  "kernel void beta_sum(global float* beta_part, global float* beta,\n",
  "                     const int x_rows, const int min_rows) {\n",
  "  const size_t col = get_global_id(0);\n",
  "  float sum = 0.0;\n",
  "  for (int i = 0; i < min_rows; i++)\n",
  "    sum += beta_part[(col * x_rows) + i];\n",
  "  beta[col] = sum;\n",
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
cl_kernel exp_kernel;
cl_kernel beta_part_kernel;
cl_kernel part_sum_kernel;
cl_kernel beta_sum_kernel;
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
      if (DEBUG) Rcout << "Device " << i << ": " << name << " (" << this_cus << " CUs) w " << mem << std::endl;
      
      if (this_cus > max_cus){
        device = devices[i];
        max_cus = this_cus;
      } // end if
    } // end for
    
    // Print out the device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    if (DEBUG) Rcout << "Using: " << name << std::endl;
    
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
    
    // Create the expectation kernel
    exp_kernel = clCreateKernel(program, "expectation", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("expectation kernel could not be created");
    }
    
    // Create the beta part kernel
    beta_part_kernel = clCreateKernel(program, "beta_part", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("beta part kernel could not be created");
    }
    
    // Create the beta part kernel
    part_sum_kernel = clCreateKernel(program, "part_sum", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("part sum kernel could not be created");
    }
    
    // Create the beta part kernel
    beta_sum_kernel = clCreateKernel(program, "beta_sum", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      stop("beta sum kernel could not be created");
    }
    
    // Don't need to reload
    kernel_loaded = true;
  } // end if
} // end load_kernel

// Releases the devices, etc used by Open CL
void release_kernel() {
  clReleaseKernel(exp_kernel);
  clReleaseKernel(beta_part_kernel);
  clReleaseKernel(part_sum_kernel);
  clReleaseKernel(beta_sum_kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  
  // Make sure to reload next time
  kernel_loaded = false;
} // end release_kernel

double f(double mu) {
  return ((R::dnorm(-mu, 0, 1, false)) / (1 - R::pnorm(-mu, 0, 1, true, false)));
} // end f

double g(double mu) {
  return ((R::dnorm(-mu, 0, 1, false)) / (R::pnorm(-mu, 0, 1, true, false)));
} // end g

void em_sequential(const arma::mat x, const arma::mat y, const arma::mat z, const int max_iter, 
                   arma::mat* beta, arma::mat* eystar) {
  // Iterations
  for (int i = 0; i < max_iter; i++) {
    arma::mat mu = x * (*beta);
    
    for (int i = 0; i < y.n_rows; i++) {
      if (y(i, 0) == 1)
        (*eystar)(i, 0) = mu(i, 0) + f(mu(i, 0));
      if (y(i, 0) == 0)
        (*eystar)(i, 0) = mu(i, 0) - g(mu(i, 0));
    } // end for
   
    // maximization step
    (*beta) = z * (*eystar);
  } // end for
} // end em_sequential

void em_parallel(arma::mat* x,  arma::mat y,  arma::mat z,  int max_iter, 
                      arma::mat* beta, arma::mat* eystar) {
  // Get the dimensions
  const int x_cols = (*x).n_cols;
  const int x_rows = (*x).n_rows;
  
  // Create float arrays for the data
  float *x_fl = new float[x_rows * x_cols];
  float *y_fl = new float[x_rows];
  float *z_fl = new float[x_cols * x_rows];
  float *beta_part_fl = new float[x_cols * x_rows];
  float *beta_fl = new float[x_cols];
  float *eystar_fl = new float[x_rows];
  
  // Copy the data to arrays
  for (int i = 0; i < x_rows; i++){
    y_fl[i] = (float)y(i, 0);
    
    for (int j = 0; j < x_cols; j++) {
      x_fl[(i * x_cols) + j] = (float)(*x)(i, j);
      z_fl[(j * x_rows) + i] = (float)z(j, i);
      beta_part_fl[(j * x_rows) + i] = 0.0;
      
      if (i == 0)
        beta_fl[j] = 0.0;
    } // end for (j)
  } // end for (i)
  if (DEBUG) warning("got here 0");
  
  // Load the OpenCL device stuff
  load_kernel();

  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
    stop("command queue could not be created");
  if (DEBUG) warning("got here 2");
    
  // Set the input memory
  cl_mem x_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (x_rows * x_cols),  x_fl, &err);
  cl_mem y_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_rows,  y_fl, &err);
  cl_mem z_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (x_cols * x_rows),  z_fl, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate input buffer");
  if (DEBUG) warning("got here 3");
  
  // Set the input/output memory
  cl_mem beta_part_io = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * (x_cols * x_rows), beta_part_fl, &err);
  cl_mem beta_io = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_cols, beta_fl, &err);
  cl_mem eystar_io = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_rows, eystar_fl, &err);
  if (err != CL_SUCCESS)
    stop("failed to allocate i/o buffer");
  if (DEBUG) warning("got here 3.5");
    
  // Set scalar memory
  const cl_int x_cols_in = x_cols;
  const cl_int x_rows_in = x_rows;
  const cl_int min_rows_in = (int)pow(2, CUTOFF);
  
  // Set the parameters
  // -- expectation
  clSetKernelArg(exp_kernel, 0, sizeof(cl_mem), &x_in);
  clSetKernelArg(exp_kernel, 1, sizeof(cl_mem), &y_in);
  clSetKernelArg(exp_kernel, 2, sizeof(cl_mem), &beta_io);
  clSetKernelArg(exp_kernel, 3, sizeof(cl_mem), &eystar_io);
  clSetKernelArg(exp_kernel, 4, sizeof(cl_int), &x_cols_in);
  clSetKernelArg(exp_kernel, 5, sizeof(cl_int), &x_rows_in);
  // -- beta part
  clSetKernelArg(beta_part_kernel, 0, sizeof(cl_mem), &z_in);
  clSetKernelArg(beta_part_kernel, 1, sizeof(cl_mem), &eystar_io);
  clSetKernelArg(beta_part_kernel, 2, sizeof(cl_mem), &beta_part_io);
  clSetKernelArg(beta_part_kernel, 3, sizeof(cl_int), &x_rows_in);
  // -- sum part
  clSetKernelArg(part_sum_kernel, 0, sizeof(cl_mem), &beta_part_io);
  clSetKernelArg(part_sum_kernel, 1, sizeof(cl_int), &x_rows_in);
  // -- beta sum
  clSetKernelArg(beta_sum_kernel, 0, sizeof(cl_mem), &beta_part_io);
  clSetKernelArg(beta_sum_kernel, 1, sizeof(cl_mem), &beta_io);
  clSetKernelArg(beta_sum_kernel, 2, sizeof(cl_int), &x_rows_in);
  clSetKernelArg(beta_sum_kernel, 3, sizeof(cl_int), &min_rows_in);
  
  if (DEBUG) warning("got here 5");
  
  // Initialize
  const int log_rows = (int)ceilf(log2f(x_rows));
  const int part_sums = log_rows - (CUTOFF - 1);
  const int exp_dim = 1;
  const int beta_part_dim = 2;
  const int part_sum_dim = 2;
  const int beta_sum_dim = 1;
  const size_t exp_dims[] = {x_rows};
  const size_t beta_part_dims[] = {x_rows, x_cols};
  const size_t beta_sum_dims[] = {x_cols};
  
  if (DEBUG) warning("got here 5.5");
  
  // Queue up the kernels for execution
  for (int i = 0; i < max_iter; i++) { 
    // expectation
    clEnqueueNDRangeKernel(queue, exp_kernel, exp_dim, NULL, exp_dims, NULL, 0, NULL, NULL);
    
    // beta = z * y*
    clEnqueueNDRangeKernel(queue, beta_part_kernel, beta_part_dim, NULL, beta_part_dims, NULL, 0, NULL, NULL);
    
    // partial sums
    for (int s = 1; s < part_sums; s++) {
      const int rows = (int)pow(2, log_rows - s);
      const size_t part_sum_dims[] = {rows, x_cols};
      clEnqueueNDRangeKernel(queue, part_sum_kernel, part_sum_dim, NULL, part_sum_dims, NULL, 0, NULL, NULL);
    } // end for
    
    // full sums
    clEnqueueNDRangeKernel(queue, beta_sum_kernel, beta_sum_dim, NULL, beta_sum_dims, NULL, 0, NULL, NULL);
  }// end for
  if (DEBUG) warning("got here 6");
  
  // Execute
  clFlush(queue);
  clFinish(queue);
  if (DEBUG) warning("got here 7");
  
  // Read out our results
  if (clEnqueueReadBuffer(queue, beta_io, CL_TRUE, 0, sizeof(float) * x_cols, beta_fl, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read out beta");
  if (clEnqueueReadBuffer(queue, eystar_io, CL_TRUE, 0, sizeof(float) * x_rows, eystar_fl, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read out eystar");
  if (clEnqueueReadBuffer(queue, beta_part_io, CL_TRUE, 0, sizeof(float) * x_cols * x_rows, beta_part_fl, 0, NULL, NULL) != CL_SUCCESS)
    stop("failed to read out beta_part");
  
  if (DEBUG) warning("got here 8");
    
  // Extract results
  for (int i = 0; i < x_cols; i++)
    (*beta)(i, 0) = beta_fl[i];
  for (int i = 0; i < x_rows; i++)
    (*eystar)(i, 0) = eystar_fl[i];

  for (int i = 0; i < x_rows; i++) {
    for (int j = 0; j < x_cols; j++) {
      if (DEBUG) Rcout << beta_part_fl[(j * x_rows) + i] << " ";
      (*x)(i,j) = (double)beta_part_fl[(j * x_rows) + i];
    }
    if (DEBUG) Rcout << std::endl;
  } 
  
  if (DEBUG) warning("got here 9");
  
  // Clean up OpenCL resources
  clReleaseMemObject(x_in);
  clReleaseMemObject(y_in);
  clReleaseMemObject(z_in);
  clReleaseMemObject(beta_part_io);
  clReleaseMemObject(beta_io);
  clReleaseMemObject(eystar_io);
    
  // Release memory
  delete [] x_fl;
  delete [] y_fl;
  delete [] z_fl;
  delete [] beta_fl;
  delete [] beta_part_fl;
  delete [] eystar_fl;
 
  // Clean up OpenCL resources (the rest)
  clReleaseCommandQueue(queue);
  release_kernel();
} // end em_parallel

// [[Rcpp::export]]
List survivalEM(const arma::mat y,  arma::mat x, // input
                const int max_iter, bool async) {
  // Check if the vectors are the same size
  if (y.n_rows != x.n_rows)
    stop("matrices not the same length");
  
  // Initialize outputs
  arma::mat beta(x.n_cols, 1);
  arma::mat eystar(x.n_rows, 1);
  beta.fill(0.0); 
  eystar.fill(0.0); 
  
  // Do some matrix stuff up front
  arma::mat z = (x.t() * x).i() * x.t();
  
  // implement algorithm
  if (async)
    em_parallel(&x, y, z, max_iter, &beta, &eystar);
  else
    em_sequential(x, y, z, max_iter, &beta, &eystar);
  
  // Output betas
  if (DEBUG) {
    for (int b = 0; b < x.n_cols; b++)
      if (async)
        Rcout << "par - beta " << b << ": " << beta(b, 0) << std::endl;
      else
        Rcout << "seq - beta " << b << ": " << beta(b, 0) << std::endl;
  } // end if
  
  // Return list
  List out;
  out["y"] = y;
  out["x"] = x;
  out["beta"] = beta;
  out["eystar"] = eystar;
  
  return out;
} // end survivalEM