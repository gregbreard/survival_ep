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
static int validate(cl_float* input1, cl_float* input2, cl_float* output) {
    int i;
    for (i = 0; i < NUM_VALUES; i++) {
        
        // The kernel was supposed to add the vectors.
        if ( output[i] != (input1[i] + input2[i]) ) {
            fprintf(stdout,
                    "Error: Element %d did not match expected output.\n", i);
            fprintf(stdout,
                    "       Saw %1.4f, expected %1.4f\n", output[i], input1[i] + input2[i]);
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
    
    // Get the platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    // Get the device count
    cl_uint deviceCount;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    
    // Get all of the devices
    cl_device_id devices[deviceCount];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    // Print the info for all the devices
    for (int i = 0; i < deviceCount; i++) {
        // Output the info
        print_device_info(devices[i]);
    } // end for
    
    // Try to obtain a dispatch queue for the GPU
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    // If no GPU available, use the CPU
    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    } // end if
    
    // This is not required, but let's print out the name of the device we're using.
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
    fprintf(stdout, "Created a dispatch queue using the %s\n", name);
    
    // Here we hardcode some test data.
    // Normally, when this application is running for real, data would come from
    // some REAL source, such as a camera, a sensor, or some compiled collection
    // of statistics—it just depends on the problem you want to solve.
    float* test_in1 = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    float* test_in2 = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    for (int i = 0; i < NUM_VALUES; i++) {
        test_in1[i] = (cl_float)i;
        test_in2[i] = (cl_float)(NUM_VALUES - i);
    } // end for
    
    // Once the computation using CL is done, will have to read the results
    // back into our application's memory space.  Allocate some space for that.
    float* test_out = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    
    // The test kernel takes two parameters: an input float array and an
    // output float array.  We can't send the application's buffers above, since
    // our CL device operates on its own memory space.  Therefore, we allocate
    // OpenCL memory for doing the work.  Notice that for the input array,
    // we specify CL_MEM_COPY_HOST_PTR and provide the fake input data we
    // created above.  This tells OpenCL to copy the data into its memory
    // space before it executes the kernel.                               // 3
    void* mem_in1  = gcl_malloc(sizeof(cl_float) * NUM_VALUES, test_in1, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* mem_in2  = gcl_malloc(sizeof(cl_float) * NUM_VALUES, test_in2, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    // The output array is not initalized; we're going to fill it up when
    // we execute our kernel.                                             // 4
    void* mem_out = gcl_malloc(sizeof(cl_float) * NUM_VALUES, NULL, CL_MEM_WRITE_ONLY);
    
    // Create the kernel
    
    // Read the program source
    std::ifstream sourceFile("ep_kernel.cl");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
    
    // Make program from the source code
    Program program=opencl::Program(context, source);
    
    // Build the program for the devices
    program.build(devices);
    
    // Make kernel
    opencl::Kernel vectorAdd_kernel(program, "vectorAdd");
    
    
    // Dispatch the kernel block using one of the dispatch_ commands and the
    // queue created earlier.                                            // 5
    
    dispatch_sync(queue, ^{
        // Although we could pass NULL as the workgroup size, which would tell
        // OpenCL to pick the one it thinks is best, we can also ask
        // OpenCL for the suggested size, and pass it ourselves.
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(vectorAdd_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        
        // The N-Dimensional Range over which we'd like to execute our
        // kernel.  In this case, we're operating on a 1D buffer, so
        // it makes sense that the range is 1D.
        cl_ndrange range = {                                              // 6
            1,                     // The number of dimensions to use.
            
            {0, 0, 0},             // The offset in each dimension.  To specify
            // that all the data is processed, this is 0
            // in the test case.                   // 7
            
            {NUM_VALUES, 0, 0},    // The global range—this is how many items
            // IN TOTAL in each dimension you want to
            // process.
            
            {wgs, 0, 0}            // The local size of each workgroup.  This
            // determines the number of work items per
            // workgroup.  It indirectly affects the
            // number of workgroups, since the global
            // size / local size yields the number of
            // workgroups.  In this test case, there are
            // NUM_VALUE / wgs workgroups.
        };
        
        // Calling the kernel is easy; simply call it like a function,
        // passing the ndrange as the first parameter, followed by the expected
        // kernel parameters.  Note that we case the 'void*' here to the
        // expected OpenCL types.  Remember, a 'float' in the
        // kernel, is a 'cl_float' from the application's perspective.   // 8
        
        vectorAdd_kernel(&range, (cl_float*)mem_in1, (cl_float*)mem_in2, (cl_float*)mem_out);
        
        // Getting data out of the device's memory space is also easy;
        // use gcl_memcpy.  In this case, gcl_memcpy takes the output
        // computed by the kernel and copies it over to the
        // application's memory space.                                   // 9
        
        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * NUM_VALUES);
        
    });
    
    // Check to see if the kernel did what it was supposed to:
    if (validate(test_in1, test_in2, test_out)) {
        fprintf(stdout, "All values were properly added.\n");
    } // end if
    
    // Clean up
    
    // Don't forget to free up the CL device's memory when you're done. // 10
    gcl_free(mem_in1);
    gcl_free(mem_in2);
    gcl_free(mem_out);
    
    // And the same goes for system memory, as usual.
    free(test_in1);
    free(test_in2);
    free(test_out);
    
    // Finally, release your queue just as you would any GCD queue.    // 11
    dispatch_release(queue);
    
    List c = List::create(a, b);
    return c;
}