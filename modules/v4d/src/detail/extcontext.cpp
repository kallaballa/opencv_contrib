// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/extcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
ExtContext::ExtContext(const int32_t& idx, cv::Ptr<FrameBufferContext> fbContext) :
        idx_(idx), mainFbContext_(fbContext), extFbContext_(new FrameBufferContext("ExtOpenGL" + std::to_string(idx), fbContext)) {
}

int ExtContext::execute(const cv::Rect& vp, std::function<void()> fn) {
		FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER, 0);
		fn();
		return 1;
}

const int32_t& ExtContext::getIndex() const {
	return idx_;
}
cv::Ptr<FrameBufferContext> ExtContext::fbCtx() {
    return extFbContext_;
}

}
}
}
