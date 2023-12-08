// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_IMGUIContext_HPP_
#define SRC_OPENCV_IMGUIContext_HPP_

#include "framebuffercontext.hpp"
#include "../../../third/imgui/imgui.h"

struct ImGuiContext;
namespace cv {
namespace v4d {
namespace detail {

class CV_EXPORTS ImGuiContextImpl {
    friend class cv::v4d::V4D;
    cv::Ptr<FrameBufferContext> mainFbContext_;
    inline static ImGuiContext* context_;
    std::function<void()> renderCallback_;
    bool firstFrame_ = true;
public:
    CV_EXPORTS ImGuiContextImpl(cv::Ptr<FrameBufferContext> fbContext);
    CV_EXPORTS void build(std::function<void()> fn);
    CV_EXPORTS static ImGuiContext* getContext();
    CV_EXPORTS static void setContext(ImGuiContext* ctx);
protected:
    CV_EXPORTS void makeCurrent();
    CV_EXPORTS void render(bool displayFPS);
};
}
}
}

#endif /* SRC_OPENCV_IMGUIContext_HPP_ */
