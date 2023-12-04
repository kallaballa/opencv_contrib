// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/nanovgcontext.hpp"
#include "opencv2/v4d/nvg.hpp"
#include "nanovg_gl.h"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(cv::Ptr<FrameBufferContext> fbContext) :
        mainFbContext_(fbContext), nvgFbContext_(new FrameBufferContext(*fbContext->getV4D(), "NanoVG", fbContext)), context_(
                nullptr) {
		FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#if defined(OPENCV_V4D_USE_ES3)
		context_ = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
#else
		context_ = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
#endif
		if (!context_)
			CV_Error(Error::StsError, "Could not initialize NanoVG!");
		nvgCreateFont(context_, "icons", "modules/v4d/assets/fonts/entypo.ttf");
		nvgCreateFont(context_, "sans", "modules/v4d/assets/fonts/Roboto-Regular.ttf");
		nvgCreateFont(context_, "sans-bold", "modules/v4d/assets/fonts/Roboto-Bold.ttf");
}

void NanoVGContext::execute(std::function<void()> fn) {
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            NanoVGContext::Scope nvgScope(*this);
            cv::v4d::nvg::detail::NVG::initializeContext(context_);
            fn();
        }
}


void NanoVGContext::begin() {
    float w = fbCtx()->size().width;
    float h = fbCtx()->size().height;
    cv::Rect vp = fbCtx()->getV4D()->plan()->viewport();
    float ws = w / vp.width;
    float hs = h / vp.height;
    float r = fbCtx()->pixelRatioX();
    CV_UNUSED(ws);
    CV_UNUSED(hs);
    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
    nvgTranslate(context_, 0, h - hs);
}

void NanoVGContext::end() {
    //FIXME make nvgCancelFrame possible

    nvgEndFrame(context_);
    nvgRestore(context_);
}

cv::Ptr<FrameBufferContext> NanoVGContext::fbCtx() {
    return nvgFbContext_;
}
}
}
}
