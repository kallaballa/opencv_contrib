// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../../include/opencv2/v4d/detail/sourcecontext.hpp"
#include "../../include/opencv2/v4d/v4d.hpp"
#include <opencv2/imgproc.hpp>

namespace cv {
namespace v4d {
namespace detail {

SourceContext::SourceContext(cv::Ptr<FrameBufferContext> mainFbContext) : mainFbContext_(mainFbContext) {
}

int SourceContext::execute(const cv::Rect& vp, std::function<void()> fn) {
	CV_UNUSED(vp);
    if (hasContext()) {
        CLExecScope_t scope(getCLExecContext());
        if (V4D::instance()->hasSource()) {
        	auto src = V4D::instance()->getSource();

        	if(src->isOpen()) {
				auto p = src->operator ()();
		        CV_Assert(p.first > 0);
		        CV_Assert(p.second.type() == CV_8UC3 || p.second.type() == CV_8UC4);

		        if(p.second.empty()) {
					CV_Error(cv::Error::StsError, "End of stream");
				}

		        if(p.second.channels() == 3)
		        	cv::cvtColor(p.second, sourceBuffer(), cv::COLOR_RGB2BGRA);
		        else
		        	p.second.copyTo(sourceBuffer());
		        fn();
		        return p.first;
        	}
        }
        return 0;
    } else {
        if (V4D::instance()->hasSource()) {
        	auto src = V4D::instance()->getSource();

        	if(src->isOpen()) {
				auto p = src->operator ()();
		        CV_Assert(p.first > 0);
		        CV_Assert(p.second.type() == CV_8UC3 || p.second.type() == CV_8UC4);

				if(p.second.empty()) {
					CV_Error(cv::Error::StsError, "End of stream");
				}

		        if(p.second.channels() == 3)
		        	cv::cvtColor(p.second, sourceBuffer(), cv::COLOR_RGB2BGRA);
		        else
		        	p.second.copyTo(sourceBuffer());
		        fn();

		        return p.first;
        	}
        }
        return 0;
    }
}

bool SourceContext::hasContext() {
    return !context_.empty();
}

void SourceContext::copyContext() {
    context_ = CLExecContext_t::getCurrent();
}

CLExecContext_t SourceContext::getCLExecContext() {
    return context_;
}

cv::UMat& SourceContext::sourceBuffer() {
	return sourceBuffer_;
}
}
}
}
