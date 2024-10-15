// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>


#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_

#if !defined(OPENCV_V4D_USE_ES3)
#    define GL_GLEXT_PROTOTYPES
#	if defined(__APPLE__)
#		include <OpenGL/gl3.h>
#	else
#		include "GL/glcorearb.h"
#	endif
#else
#	include "GLES3/gl3.h"
#	include "GLES2/gl2ext.h"
#endif

#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_GL_HPP_ */
