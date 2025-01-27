// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../../include/opencv2/v4d/detail/sinkcontext.hpp"
#include "../../include/opencv2/v4d/v4d.hpp"

#include <opencv2/imgproc.hpp>

namespace cv {
namespace v4d {
namespace detail {

SinkContext::SinkContext(cv::Ptr<FrameBufferContext> mainFbContext) : mainFbContext_(mainFbContext) {
}

int SinkContext::execute(const cv::Rect& vp, std::function<void()> fn) {
	if(V4D::instance()->get<bool>(V4D::Keys::DISABLE_VIDEO_IO))
		return 1;

	CV_UNUSED(vp);
    if (hasContext()) {
        CLExecScope_t scope(getCLExecContext());
        fn();
    } else {
    	fn();
    }
	auto v4d = V4D::instance();
	if(v4d->hasSink() && v4d->getSink()->isOpen()) {
			cvtColor(sinkBuffer(), rgba_, cv::COLOR_BGRA2RGBA);
			v4d->getSink()->operator ()(v4d->getSequenceNumber(), rgba_);
			return 1;
	}
	return 0;
}

bool SinkContext::hasContext() {
    return !context_.empty();
}

void SinkContext::copyContext() {
    context_ = CLExecContext_t::getCurrent();
}

CLExecContext_t SinkContext::getCLExecContext() {
    return context_;
}

cv::UMat& SinkContext::sinkBuffer() {
	return sinkBuffer_;
}
}
}
}
