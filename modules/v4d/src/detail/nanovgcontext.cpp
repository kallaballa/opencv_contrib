// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/gl.hpp"
#include "opencv2/v4d/detail/nanovgcontext.hpp"
#include "opencv2/v4d/nvg.hpp"
#include "nanovg_gl.h"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(cv::Ptr<FrameBufferContext> fbContext) :
        mainFbContext_(fbContext), nvgFbContext_(new FrameBufferContext("NanoVG", fbContext)), context_(
                nullptr) {
		FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#if defined(OPENCV_V4D_USE_ES3)
		context_ = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
#else
		context_ = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
#endif
		if (!context_)
			CV_Error(Error::StsError, "Could not initialize NanoVG!");
		nvgCreateFont(context_, "icons", "modules/v4d/assets/fonts/entypo.ttf");
		nvgCreateFont(context_, "sans", "modules/v4d/assets/fonts/Roboto-Regular.ttf");
		nvgCreateFont(context_, "sans-bold", "modules/v4d/assets/fonts/Roboto-Bold.ttf");
}

int NanoVGContext::execute(const cv::Rect& vp, std::function<void()> fn) {
	{
		FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
		glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glViewport(vp.x, vp.y, vp.width, vp.height);
		NanoVGContext::Scope nvgScope(*this, vp);
		cv::v4d::nvg::detail::NVG::initializeContext(context_);
		fn();
		return 1;
	}
}


void NanoVGContext::begin(const cv::Rect& viewport) {
    float w = fbCtx()->size().width;
    float h = fbCtx()->size().height;
    float ws = viewport.width;
    float hs = viewport.height;
    float r = fbCtx()->pixelRatioX();
    CV_UNUSED(ws);
    nvgSave(context_);
    nvgBeginFrame(context_, ws, hs, r);
//    nvgTranslate(context_, 0, h - hs);
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
