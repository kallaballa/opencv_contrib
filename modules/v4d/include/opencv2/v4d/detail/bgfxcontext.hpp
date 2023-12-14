// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_BGFXGCONTEXT_HPP_
#define SRC_OPENCV_BGFXGCONTEXT_HPP_

#define BX_CONFIG_DEBUG 0

#include "framebuffercontext.hpp"
#include "bgfx/bgfx.h"
#include "bx/bx.h"
#include "bx/allocator.h"
#include "bx/file.h"
#include "bx/timer.h"

namespace cv {
namespace v4d {
namespace util {
//adapted from https://github.com/bkaradzic/bgfx/blob/07be0f213acd73a4f6845dc8f7b20b93f66b7cc4/examples/common/bgfx_utils.cpp
static thread_local bx::DefaultAllocator allocator;
static thread_local bx::FileReaderI* file_reader = BX_NEW(&allocator, bx::FileReader);;
static const bgfx::Memory* load_mem(bx::FileReaderI* _reader, const char* _filePath)
{
	if (bx::open(_reader, _filePath) )
	{
		uint32_t size = (uint32_t)bx::getSize(_reader);
		const bgfx::Memory* mem = bgfx::alloc(size+1);
		bx::read(_reader, mem->data, size, bx::ErrorAssert{});
		bx::close(_reader);
		mem->data[mem->size-1] = '\0';
		return mem;
	}

	std::cerr << "Failed to load" << _filePath << std::endl;
	return NULL;
}

static bgfx::ShaderHandle load_shader(bx::FileReaderI* _reader, const char* _name)
{
	char filePath[512];

	const char* shaderPath = "???";

	switch (bgfx::getRendererType() )
	{
	case bgfx::RendererType::Noop:
	case bgfx::RendererType::Direct3D11:
	case bgfx::RendererType::Direct3D12: shaderPath = "shaders/dx11/";  break;
	case bgfx::RendererType::Agc:
	case bgfx::RendererType::Gnm:        shaderPath = "shaders/pssl/";  break;
	case bgfx::RendererType::Metal:      shaderPath = "shaders/metal/"; break;
	case bgfx::RendererType::Nvn:        shaderPath = "shaders/nvn/";   break;
	case bgfx::RendererType::OpenGL:     shaderPath = "shaders/glsl/";  break;
	case bgfx::RendererType::OpenGLES:   shaderPath = "shaders/essl/";  break;
	case bgfx::RendererType::Vulkan:     shaderPath = "shaders/spirv/"; break;

	case bgfx::RendererType::Count:
		BX_ASSERT(false, "You should not be here!");
		break;
	}

	bx::strCopy(filePath, BX_COUNTOF(filePath), shaderPath);
	bx::strCat(filePath, BX_COUNTOF(filePath), _name);
	bx::strCat(filePath, BX_COUNTOF(filePath), ".bin");

	bgfx::ShaderHandle handle = bgfx::createShader(load_mem(_reader, filePath));
	bgfx::setName(handle, _name);

	return handle;
}

static const bgfx::Memory* load_mem(bx::FileReaderI* _reader, const char* _filePath);
static bgfx::ShaderHandle load_shader(bx::FileReaderI* _reader, const char* _name);
bgfx::ShaderHandle load_shader(const char* _name);
bgfx::ProgramHandle load_program(bx::FileReaderI* _reader, const char* _vsName, const char* _fsName);
bgfx::ProgramHandle load_program(const char* _vsName, const char* _fsName);
}
namespace detail {

class CV_EXPORTS BgfxContext : public V4DContext {
	cv::Ptr<FrameBufferContext> mainFbContext_;
	cv::Ptr<FrameBufferContext> rayFbContext_;
public:
    BgfxContext(cv::Ptr<FrameBufferContext> fbContext);
    virtual ~BgfxContext() {};

    virtual int execute(const cv::Rect& vp, std::function<void()> fn) override;

    cv::Ptr<FrameBufferContext> fbCtx();
};
}
}
}

#endif /* SRC_OPENCV_BGFXGCONTEXT_HPP_ */
