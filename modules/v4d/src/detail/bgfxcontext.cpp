// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../../include/opencv2/v4d/detail/bgfxcontext.hpp"
#include "../../include/opencv2/v4d/v4d.hpp"

#include <bx/bx.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>
#if BX_PLATFORM_LINUX
#define GLFW_EXPOSE_NATIVE_X11
#elif BX_PLATFORM_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#elif BX_PLATFORM_OSX
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3native.h>


namespace cv {
namespace v4d {
namespace util {
CV_EXPORTS bgfx::ShaderHandle load_shader(const char* _name)
{
	return load_shader(file_reader, _name);
}

CV_EXPORTS bgfx::ProgramHandle load_program(bx::FileReaderI* _reader, const char* _vsName, const char* _fsName)
{
	bgfx::ShaderHandle vsh = load_shader(_reader, _vsName);
	bgfx::ShaderHandle fsh = BGFX_INVALID_HANDLE;
	if (NULL != _fsName)
	{
		fsh = load_shader(_reader, _fsName);
	}

	return bgfx::createProgram(vsh, fsh, true /* destroy shaders when program is destroyed */);
}

CV_EXPORTS bgfx::ProgramHandle load_program(const char* _vsName, const char* _fsName)
{
	return load_program(file_reader, _vsName, _fsName);
}
}
namespace detail {

BgfxContext::BgfxContext(cv::Ptr<FrameBufferContext> fbContext) :
	mainFbContext_(fbContext),
	bgfxContext_(FrameBufferContext::make("Bgfx", fbContext)) {
//	bgfx::renderFrame();
	bgfx::Init init;
#ifndef OPENCV_V4D_USE_ES3
	init.type     = bgfx::RendererType::OpenGL;
#else
	init.type     = bgfx::RendererType::OpenGLES;
#endif
#if BX_PLATFORM_LINUX || BX_PLATFORM_BSD
	init.platformData.ndt = glfwGetX11Display();
	init.platformData.nwh = (void*)(uintptr_t)glfwGetX11Window(fbCtx()->getGLFWWindow());
#elif BX_PLATFORM_OSX
	init.platformData.nwh = glfwGetCocoaWindow(fbCtx()->getGLFWWindow());
#elif BX_PLATFORM_WINDOWS
	init.platformData.nwh = glfwGetWin32Window(fbCtx()->getGLFWWindow());
#endif
	cv::Size sz = fbCtx()->size();
	init.resolution.width  = sz.width;
	init.resolution.height = sz.height;
	init.resolution.reset  = BGFX_RESET_VSYNC;
	FrameBufferContext::WindowScope winScope(fbCtx());
	FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER, 0, true);
	bgfx::init(init);

	// Enable debug text.
	bgfx::setDebug(BGFX_DEBUG_NONE);
}

int BgfxContext::execute(const cv::Rect& vp, std::function<void()> fn) {
	FrameBufferContext::WindowScope winScope(fbCtx());
	FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER, 0, true);
	fn();
	CV_Assert(fbCtx()->getGLFWWindow() == glfwGetCurrentContext());
	return 1;
}


cv::Ptr<FrameBufferContext> BgfxContext::fbCtx() {
    return bgfxContext_;
}
}
}
}
