// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_NANOVGCONTEXT_HPP_
#define SRC_OPENCV_NANOVGCONTEXT_HPP_

#include "framebuffercontext.hpp"

struct NVGcontext;
namespace cv {
namespace v4d {
namespace detail {
/*!
 * Used to setup a nanovg context
 */
class CV_EXPORTS NanoVGContext : public V4DContext {
	cv::Ptr<FrameBufferContext> mainFbContext_;
	cv::Ptr<FrameBufferContext> nvgFbContext_;
    NVGcontext* context_;
public:
    /*!
     * Makes sure #NanoVGContext::begin and #NanoVGContext::end are both called
     */
    class Scope {
        NanoVGContext& ctx_;
    public:
        /*!
         * Setup NanoVG rendering
         * @param ctx The corresponding #NanoVGContext
         */
        Scope(NanoVGContext& ctx, const cv::Rect& viewport) :
                ctx_(ctx) {
            ctx_.begin(viewport);
        }
        /*!
         * Tear-down NanoVG rendering
         */
        ~Scope() {
            ctx_.end();
        }
    };

    /*!
     * Creates a NanoVGContext
     * @param v4d The V4D object used in conjunction with this context
     * @param context The native NVGContext
     * @param fbContext The framebuffer context
     */
    NanoVGContext(cv::Ptr<FrameBufferContext> fbContext);
    virtual ~NanoVGContext() {};

    /*!
     * Execute function object fn inside a nanovg context.
     * The context takes care of setting up opengl and nanovg states.
     * A function object passed like that can use the functions in cv::viz::nvg.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::viz::nvg
     */
    virtual int execute(const cv::Rect& vp, std::function<void()> fn) override;

    cv::Ptr<FrameBufferContext> fbCtx();
private:
    /*!
     * Setup NanoVG context
     */
    void begin(const cv::Rect& viewport);
    /*!
     * Tear down NanoVG context
     */
    void end();

};
}
}
}

#endif /* SRC_OPENCV_NANOVGCONTEXT_HPP_ */
