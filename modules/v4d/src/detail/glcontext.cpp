// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/glcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
GLContext::GLContext(const int32_t& idx, cv::Ptr<FrameBufferContext> fbContext) :
        idx_(idx), mainFbContext_(fbContext), glFbContext_(new FrameBufferContext(*fbContext->getV4D(), "OpenGL" + std::to_string(idx), fbContext)) {
}

void GLContext::execute(std::function<void()> fn) {
		FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
		fn();
}

const int32_t& GLContext::getIndex() const {
	return idx_;
}

cv::Ptr<FrameBufferContext> GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
