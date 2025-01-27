#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_CL_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_CL_HPP_

#ifdef HAVE_OPENCL

#ifndef CL_TARGET_OPENCL_VERSION
#  define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#  include <OpenCL/cl_gl_ext.h>
#else
#  include <CL/cl_gl.h>
#endif

#endif

#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_CL_HPP_ */
