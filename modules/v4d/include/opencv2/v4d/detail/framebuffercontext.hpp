// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_
#define SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_

#include "context.hpp"
#include "cl.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <map>
#include <vector>
#include <string>

typedef unsigned int GLenum;
using std::string;
#define GL_FRAMEBUFFER 0x8D40

struct GLFWwindow;
namespace cv {
namespace v4d {
class V4D;
class Plan;

namespace detail {
struct FBConfigFlags {
	enum Enum {
		NONE = 0,
		OFFSCREEN = 1,
		DEBUG_GL_CONTEXT = 2,
		ONSCREEN_CHILD_CONTEXTS = 4,
		VSYNC = 8,
		DISPLAY_MODE = 16,
		DEFAULT = NONE
	};
};

#ifdef HAVE_OPENCL
typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
class CLExecScope_t
{
    CLExecContext_t ctx_;
public:
    inline CLExecScope_t(const CLExecContext_t& ctx)
    {
    	if(cv::ocl::useOpenCL()) {
			CV_Assert(!ctx.empty());
			ctx_ = CLExecContext_t::getCurrentRef();
			ctx.bind();
		}
    }

    inline ~CLExecScope_t()
    {
    	if(cv::ocl::useOpenCL()) {
			if (!ctx_.empty())
			{
				ctx_.bind();
			}
    	}
    }
};
#else
struct CLExecContext_t {
	bool empty() {
		return true;
	}
	static CLExecContext_t getCurrent() {
		return CLExecContext_t();
	}
};
class CLExecScope_t
{
    CLExecContext_t ctx_;
public:
    inline CLExecScope_t(const CLExecContext_t& ctx)
    {
    }

    inline ~CLExecScope_t()
    {
    }
};
#endif
/*!
 * The FrameBufferContext acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
 */
class CV_EXPORTS FrameBufferContext : public V4DContext {
    typedef unsigned int GLuint;
    typedef signed int GLint;

    friend class SourceContext;
    friend class SinkContext;
    friend class GLContext;
    friend class ExtContext;
    friend class NanoVGContext;
    friend class ImGuiContextImpl;
    friend class BgfxContext;
    friend class cv::v4d::V4D;
    friend class cv::v4d::Plan;
    cv::Ptr<FrameBufferContext> self_;
    string title_;
    int major_;
    int minor_;
    int samples_;
    int configFlags_;
    bool isVisible_;
    GLFWwindow* glfwWindow_ = nullptr;
    GLFWwindow* currentContext_ = nullptr;
    bool clglSharing_ = true;
    GLuint onscreenTextureID_ = 0;
    GLuint onscreenRenderBufferID_ = 0;
    GLuint framebufferID_ = 0;
    GLuint framebufferFlippedID_ = 0;
    GLuint textureFlippedID_ = 0;
    GLuint textureID_ = 0;
    GLuint renderBufferID_ = 0;
    cl_mem clImage_ = nullptr;
    CLExecContext_t context_;
    const cv::Size framebufferSize_;
    cv::Ptr<FrameBufferContext> parent_;
    bool isRoot_ = true;
    int index_;
    std::map<size_t, GLint> texture_hdls_;
    std::map<size_t, GLuint> shader_program_hdls_;
    std::map<size_t, GLuint> copyFramebuffers_;
    std::map<size_t, GLuint> copyTextures_;

    /*!
     * Create a FrameBufferContext with given size.
     * @param frameBufferSize The frame buffer size.
     */
public:

    /*!
     * Acquires and releases the framebuffer from and to OpenGL.
     */
    class CV_EXPORTS FrameBufferScope {
    	cv::Ptr<FrameBufferContext> ctx_;
        cv::UMat& m_;
#ifdef HAVE_OPENCL
        std::shared_ptr<CLExecContext_t> pExecCtx;
#endif
    public:
        /*!
         * Aquires the framebuffer via cl-gl sharing.
         * @param ctx The corresponding #FrameBufferContext.
         * @param m The UMat to bind the OpenGL framebuffer to.
         */
        CV_EXPORTS FrameBufferScope(cv::Ptr<FrameBufferContext> ctx, cv::UMat& m) :
                ctx_(ctx), m_(m)
#ifdef HAVE_OPENCL
        , pExecCtx(std::static_pointer_cast<CLExecContext_t>(m.u->allocatorContext))
#endif
        {
            CV_Assert(!m.empty());
#ifdef HAVE_OPENCL
            if(pExecCtx) {
                CLExecScope_t execScope(*pExecCtx.get());
                ctx_->acquireFromGL(m_);
            } else {
#endif
                ctx_->acquireFromGL(m_);
#ifdef HAVE_OPENCL
            }
#endif
        }
        /*!
         * Releases the framebuffer via cl-gl sharing.
         */
        CV_EXPORTS virtual ~FrameBufferScope() {
#ifdef HAVE_OPENCL
            if (pExecCtx) {
                CLExecScope_t execScope(*pExecCtx.get());
                ctx_->releaseToGL(m_);
            }
            else {
#endif
                ctx_->releaseToGL(m_);
#ifdef HAVE_OPENCL
            }
#endif
        }
    };

    /*!
     * Setups and tears-down OpenGL states.
     */
    class CV_EXPORTS GLScope {
    	cv::Ptr<FrameBufferContext> ctx_;
    	bool copyBack_;
    public:
        /*!
         * Setup OpenGL states.
         * @param ctx The corresponding #FrameBufferContext.
         */
        CV_EXPORTS GLScope(cv::Ptr<FrameBufferContext> ctx, GLenum framebufferTarget, GLint frameBufferID = -1, bool copyBack = false) :
			ctx_(ctx),
			copyBack_(copyBack) {
        	CV_Assert(ctx);
        	if(frameBufferID == -1)
				frameBufferID = ctx->framebufferID_;
            ctx_->begin(framebufferTarget, frameBufferID);
        }
        /*!
         * Tear-down OpenGL states.
         */
        CV_EXPORTS ~GLScope() {
            ctx_->end(copyBack_);
        }
    };

    class CV_EXPORTS WindowScope {
    	cv::Ptr<FrameBufferContext> ctx_;
    public:
        CV_EXPORTS WindowScope(cv::Ptr<FrameBufferContext> ctx) :
			ctx_(ctx) {
        	CV_Assert(ctx_);
        	ctx_->makeCurrent();
        }

        CV_EXPORTS ~WindowScope() {
        	ctx_->makeNoneCurrent();
        }
    };

private:
    FrameBufferContext(const cv::Size& frameBufferSize, const string& title, int major, int minor, int samples, GLFWwindow* rootWindow, cv::Ptr<FrameBufferContext> parent, bool root, int configFlags = -1);
    FrameBufferContext(const string& title, cv::Ptr<FrameBufferContext> other);
public:
    virtual ~FrameBufferContext();

    static cv::Ptr<FrameBufferContext> make(const string& title, cv::Ptr<FrameBufferContext> other);
    static cv::Ptr<FrameBufferContext> make(const cv::Size& sz, const string& title, const int& major, const int& minor, const int& samples, GLFWwindow* parentWindow, cv::Ptr<FrameBufferContext> parent, const bool& root, const int& confFlags = -1);


    cv::Ptr<FrameBufferContext> self() {
    	return self_;
    }



    GLuint getFramebufferID();
    GLuint getTextureID();

    /*!
     * Get the framebuffer size.
     * @return The framebuffer size.
     */
    const cv::Size& size() const;
    void copyTo(cv::UMat& dst);
    void copyFrom(const cv::UMat& src);
    void copyToRootWindow();

    /*!
      * Execute function object fn inside a framebuffer context.
      * The context acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
      * and provides it to the functon object. This is a good place to use OpenCL
      * directly on the framebuffer.
      * @param fn A function object that is passed the framebuffer to be read/manipulated.
      */
    virtual int execute(const cv::Rect& vp, std::function<void()> fn) override {
		cv::Rect glAdjustedVp(0, vp.x + (size().height - vp.height), vp.width, vp.height);
    	FrameBufferContext::WindowScope winScope(self());
    	if(cv::ocl::useOpenCL() && !getCLExecContext().empty()) {
			CLExecScope_t clExecScope(getCLExecContext());
			FrameBufferContext::GLScope glScope(self(), GL_FRAMEBUFFER);
			FrameBufferContext::FrameBufferScope fbScope(self(), framebuffer_);

			view_ = framebuffer_(glAdjustedVp);
			fn();
    	} else {
			FrameBufferContext::GLScope glScope(self(), GL_FRAMEBUFFER);
			FrameBufferContext::FrameBufferScope fbScope(self(), framebuffer_);
			view_ = framebuffer_(glAdjustedVp);
			fn();
    	}

    	return 1;
    }

    cv::Vec2f position();
    float pixelRatioX();
    float pixelRatioY();
    void makeCurrent();
    void makeNoneCurrent();
    bool isResizable();
    void setResizable(bool r);
    void setWindowSize(const cv::Size& sz);
    bool isFullscreen();
    void setFullscreen(bool f);
    cv::Size getNativeFrameBufferSize();
    void setVisible(bool v);
    bool isVisible();
    void close();
    bool isClosed();
    bool isRoot();
    bool hasParent();

    /*!
     * Blit the framebuffer to the screen
     * @param viewport ROI to blit
     * @param windowSize The size of the window to blit to
     * @param stretch if true stretch the framebuffer to window size
     */
    void blitFrameBufferToFrameBuffer(const cv::Rect& srcViewport, const cv::Size& targetFbSize,
            GLuint targetFramebufferID = 0, bool stretch = true, bool flipY = false);
protected:
    CLExecContext_t& getCLExecContext();
    int getIndex();
    void setup();
    void teardown();
    void flip();
    void unflip();
    GLFWwindow* getGLFWWindow() const;
public:
    CV_EXPORTS int configFlags();
    CV_EXPORTS void loadShaders(const size_t& index);
    CV_EXPORTS void initBlend(const size_t& index);
    CV_EXPORTS void blendFramebuffer(const GLuint& otherID);
    CV_EXPORTS void init();
    CV_EXPORTS cv::UMat& fb();
    /*!
     * Setup OpenGL states.
     */
    CV_EXPORTS void begin(GLenum framebufferTarget, GLuint frameBufferID);
    /*!
     * Tear-down OpenGL states.
     */
    CV_EXPORTS void end(bool copyBack);
    /*!
     * Download the framebuffer to UMat m.
     * @param m The target UMat.
     */
    void download(cv::UMat& m);
    /*!
     * Uploat UMat m to the framebuffer.
     * @param m The UMat to upload.
     */
    void upload(const cv::UMat& m);
    /*!
     * Acquire the framebuffer using cl-gl sharing.
     * @param m The UMat the framebuffer will be bound to.
     */
    void acquireFromGL(cv::UMat& m);
    /*!
     * Release the framebuffer using cl-gl sharing.
     * @param m The UMat the framebuffer is bound to.
     */
    void releaseToGL(cv::UMat& m);
    void toGLTexture2D(cv::UMat& u, const GLuint& texID);
    void fromGLTexture2D(const GLuint& texID, cv::UMat& u);

    template<typename _Tp>
    struct RectLessCompare
    {
        bool operator()(const cv::Rect_<_Tp>& lhs, const cv::Rect_<_Tp>& rhs) const {
        	if(lhs.x != rhs.x)
        		return lhs.x < rhs.x;
        	else if(lhs.y != rhs.y)
        		return lhs.y < rhs.y;
        	else if(lhs.width != rhs.width)
        		return lhs.width < rhs.width;
        	else
        		return lhs.height < rhs.height;
        }
    };

    cv::UMat framebuffer_;
    cv::UMat view_;
    GLint currentFBO_ = -1;
    GLint currentFBOTarget_ = -1;
};
}
}
}

#endif /* SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_ */
