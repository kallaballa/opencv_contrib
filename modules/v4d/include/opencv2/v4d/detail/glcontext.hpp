// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_GLCONTEXT_HPP_
#define SRC_OPENCV_GLCONTEXT_HPP_

#include "opencv2/v4d/detail/framebuffercontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
/*!
 * Used to setup an OpengLG context
 */
class CV_EXPORTS GLContext : public V4DContext {
	const int32_t idx_;
    cv::Ptr<FrameBufferContext> mainFbContext_;
    cv::Ptr<FrameBufferContext> glFbContext_;
public:
    GLContext(const int32_t& idx, cv::Ptr<FrameBufferContext> fbContext);
    virtual ~GLContext() {};
    virtual void execute(std::function<void()> fn) override;
    const int32_t& getIndex() const;
    cv::Ptr<FrameBufferContext> fbCtx();
};
}
}
}

#endif /* SRC_OPENCV_GLCONTEXT_HPP_ */
