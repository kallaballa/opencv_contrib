// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <sstream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>
#include <semaphore>

#include "../include/opencv2/v4d/v4d.hpp"
#include "../include/opencv2/v4d/detail/framebuffercontext.hpp"
#include "../include/opencv2/v4d/detail/gl.hpp"

namespace cv {
namespace v4d {

CV_EXPORTS thread_local std::mutex V4D::instance_mtx_;
CV_EXPORTS thread_local cv::Ptr<V4D> V4D::instance_;
CV_EXPORTS ThreadSafeAnyMap<V4D::Keys::Enum> V4D::properties_;

cv::Ptr<V4D> V4D::init(const cv::Rect& viewport, const string& title, int allocFlags, int confFlags, int debFlags, int samples) {
	std::lock_guard guard(instance_mtx_);
	instance_ = new V4D(viewport, cv::Size(), title, allocFlags, confFlags, debFlags, samples);
	return instance_;
}

cv::Ptr<V4D> V4D::init(const cv::Rect& viewport, const cv::Size& fbsize, const string& title, int allocFlags, int confFlags, int debFlags, int samples) {
	std::lock_guard guard(instance_mtx_);
	instance_ = new V4D(viewport, fbsize, title, allocFlags, confFlags, debFlags, samples);
	return instance_;
}

cv::Ptr<V4D> V4D::init(const V4D& other, const string& title) {
	std::lock_guard guard(instance_mtx_);
	instance_ = new V4D(other, title);
	return instance_;
}

V4D::V4D(const cv::Rect& viewport, cv::Size fbsize, const string& title, int allocFlags, int confFlags, int debFlags, int samples) :
        allocateFlags_(allocFlags), configFlags_(confFlags), debugFlags_(debFlags), samples_(samples) {
	if(fbsize.empty())
    	fbsize = viewport.size();
	create<true>(Keys::INIT_VIEWPORT, viewport);
    create<false>(Keys::VIEWPORT, viewport);
    create<false, cv::Size>(Keys::WINDOW_SIZE, viewport.size(), [this](const cv::Size& sz){ fbCtx()->setWindowSize(sz); });
	create<true>(Keys::FB_SIZE, fbsize);
    create<false>(Keys::STRETCHING, true);
    create<false>(Keys::CLEAR_COLOR, cv::Scalar(0, 0, 0, 255));
    create<false,string>(Keys::NAMESPACE, "default");
    create<false, bool>(Keys::FULLSCREEN, false, [this](const bool& b){ fbCtx()->setFullscreen(b); });

    int fbFlags = FBConfigFlags::VSYNC
    		| (debugFlags() &  DebugFlags::DEBUG_GL_CONTEXT ? FBConfigFlags::DEBUG_GL_CONTEXT : 0)
			| (debugFlags() &  DebugFlags::ONSCREEN_CONTEXTS ? FBConfigFlags::ONSCREEN_CHILD_CONTEXTS : 0)
			| (configFlags() &  ConfigFlags::OFFSCREEN ? FBConfigFlags::OFFSCREEN : 0);
    mainFbContext_ = new detail::FrameBufferContext(fbsize.empty() ? viewport.size() : fbsize, title, 3,
                2, samples, nullptr, nullptr, true, fbFlags);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);

    if(allocateFlags() & AllocateFlags::IMGUI)
        imguiContext_ = new detail::ImGuiContextImpl(mainFbContext_);

    setVisible(!(configFlags() & ConfigFlags::OFFSCREEN));
}

V4D::V4D(const V4D& other, const string& title) :
		allocateFlags_(other.allocateFlags_), configFlags_(other.configFlags_), debugFlags_(other.debugFlags_), samples_(other.samples_) {
	workerIdx_ = RunState::instance().apply<size_t>(RunState::Keys::WORKERS_INDEX, [](size_t& v){ return v++; });
    int fbFlags = (configFlags() &  ConfigFlags::DISPLAY_MODE ? FBConfigFlags::DISPLAY_MODE : 0)
    		| (debugFlags() &  DebugFlags::DEBUG_GL_CONTEXT ? FBConfigFlags::DEBUG_GL_CONTEXT : 0)
			| (debugFlags() &  DebugFlags::ONSCREEN_CONTEXTS ? FBConfigFlags::ONSCREEN_CHILD_CONTEXTS : FBConfigFlags::OFFSCREEN);

    mainFbContext_ = new detail::FrameBufferContext(other.get<cv::Size>(V4D::Keys::FB_SIZE), title, 3,
                2, other.samples_, other.fbCtx()->rootWindow_, other.fbCtx(), true, fbFlags);

    CLExecScope_t scope(mainFbContext_->getCLExecContext());
    if(allocateFlags() & AllocateFlags::NANOVG)
    	nvgContext_ = new detail::NanoVGContext(mainFbContext_);
    if(allocateFlags() & AllocateFlags::BGFX)
        bgfxContext_ = new detail::BgfxContext(mainFbContext_);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);
    plainContext_ = new detail::PlainContext();

    setVisible(debugFlags() & DebugFlags::ONSCREEN_CONTEXTS);
}

V4D::~V4D() {

}

const int32_t& V4D::workerIndex() const {
	return workerIdx_;
}

std::string V4D::title() const {
    return fbCtx()->title_;
}

cv::Ptr<FrameBufferContext> V4D::fbCtx() const {
    assert(mainFbContext_ != nullptr);
    return mainFbContext_;
}

cv::Ptr<SourceContext> V4D::sourceCtx() {
    assert(sourceContext_ != nullptr);
    return sourceContext_;
}

cv::Ptr<SinkContext> V4D::sinkCtx() {
    assert(sinkContext_ != nullptr);
    return sinkContext_;
}

cv::Ptr<NanoVGContext> V4D::nvgCtx() {
    assert(nvgContext_ != nullptr);
    return nvgContext_;
}

cv::Ptr<BgfxContext> V4D::bgfxCtx() {
    assert(bgfxContext_ != nullptr);
    return bgfxContext_;
}

cv::Ptr<PlainContext> V4D::plainCtx() {
    assert(plainContext_ != nullptr);
    return plainContext_;
}

cv::Ptr<ImGuiContextImpl> V4D::imguiCtx() {
    assert(imguiContext_ != nullptr);
    return imguiContext_;
}

cv::Ptr<GLContext> V4D::glCtx(int32_t idx) {
    auto it = glContexts_.find(idx);
    if(it != glContexts_.end())
        return (*it).second;
    else {
        cv::Ptr<GLContext> ctx = new GLContext(idx, mainFbContext_);
        glContexts_.insert({idx, ctx});
        return ctx;
    }
}

cv::Ptr<ExtContext> V4D::extCtx(int32_t idx) {
    auto it = extContexts_.find(idx);
    if(it != extContexts_.end())
        return (*it).second;
    else {
        cv::Ptr<ExtContext> ctx = new ExtContext(idx, mainFbContext_);
        extContexts_.insert({idx, ctx});
        return ctx;
    }
}

bool V4D::hasFbCtx() {
    return mainFbContext_ != nullptr;
}

bool V4D::hasSourceCtx() {
    return sourceContext_ != nullptr;
}

bool V4D::hasSinkCtx() {
    return sinkContext_ != nullptr;
}

bool V4D::hasNvgCtx() {
    return nvgContext_ != nullptr;
}

bool V4D::hasBgfxCtx() {
    return bgfxContext_ != nullptr;
}

bool V4D::hasPlainCtx() {
    return plainContext_ != nullptr;
}

bool V4D::hasImguiCtx() {
    return imguiContext_ != nullptr;
}

bool V4D::hasGlCtx(uint32_t idx) {
    return glContexts_.find(idx) != glContexts_.end();
}

bool V4D::hasExtCtx(uint32_t idx) {
    return extContexts_.find(idx) != extContexts_.end();
}

size_t V4D::numGlCtx() {
    return std::max(off_t(0), off_t(glContexts_.size()) - 1);
}

size_t V4D::numExtCtx() {
    return std::max(off_t(0), off_t(extContexts_.size()) - 1);
}

void V4D::copyTo(cv::UMat& m) {
	fbCtx()->copyTo(m);
}

void V4D::copyFrom(const cv::UMat& m) {
	fbCtx()->copyFrom(m);
}

void V4D::setSource(cv::Ptr<Source> src) {
    source_ = src;
}
cv::Ptr<Source> V4D::getSource() {
    return source_;
}

bool V4D::hasSource() const {
    return source_ != nullptr;
}

void V4D::setSink(cv::Ptr<Sink> sink) {
    sink_ = sink;
}

cv::Ptr<Sink> V4D::getSink() {
    return sink_;
}

bool V4D::hasSink() const {
    return sink_ != nullptr;
}

cv::Vec2f V4D::position() {
    return fbCtx()->position();
}

float V4D::pixelRatioX() {
    return fbCtx()->pixelRatioX();
}

float V4D::pixelRatioY() {
    return fbCtx()->pixelRatioY();
}

const cv::Size& V4D::size() {
    return get<cv::Size>(Keys::WINDOW_SIZE);
}

void V4D::setShowFPS(bool s) {
    showFPS_ = s;
}

bool V4D::getShowFPS() {
    return showFPS_;
}

void V4D::setPrintFPS(bool p) {
    printFPS_ = p;
}

bool V4D::getPrintFPS() {
    return printFPS_;
}

void V4D::setShowTracking(bool st) {
    showTracking_ = st;
}

bool V4D::getShowTracking() {
    return showTracking_;
}

bool V4D::isFullscreen() {
    return fbCtx()->isFullscreen();
}

void V4D::setFullscreen(bool f) {
    fbCtx()->setFullscreen(f);
}

bool V4D::isResizable() {
    return fbCtx()->isResizable();
}

void V4D::setResizable(bool r) {
    fbCtx()->setResizable(r);
}

bool V4D::isVisible() {
    return fbCtx()->isVisible();
}

void V4D::setVisible(bool v) {
    fbCtx()->setVisible(v);
}

void V4D::swapContextBuffers() {
	cv::Rect fbViewport(0, 0, fbCtx()->size().width, fbCtx()->size().height);
    for(int32_t i = -1; i < numGlCtx(); ++i) {
        FrameBufferContext::GLScope glScope(glCtx(i)->fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx(i)->fbCtx()->blitFrameBufferToFrameBuffer(fbViewport, size(), 0, get<bool>(Keys::STRETCHING));
//        GL_CHECK(glFinish());
        glfwSwapBuffers(glCtx(i)->fbCtx()->getGLFWWindow());
    }

    if(hasNvgCtx()) {
		FrameBufferContext::GLScope glScope(nvgCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
		nvgCtx()->fbCtx()->blitFrameBufferToFrameBuffer(fbViewport, size(), 0, get<bool>(Keys::STRETCHING));
//        GL_CHECK(glFinish());
		glfwSwapBuffers(nvgCtx()->fbCtx()->getGLFWWindow());
    }

    if(hasBgfxCtx()) {
		FrameBufferContext::GLScope glScope(bgfxCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
		bgfxCtx()->fbCtx()->blitFrameBufferToFrameBuffer(fbViewport, size(), 0, get<bool>(Keys::STRETCHING));
//        GL_CHECK(glFinish());
		glfwSwapBuffers(bgfxCtx()->fbCtx()->getGLFWWindow());
    }

}

bool V4D::display() {
	Global& global = Global::instance();
	RunState& state = RunState::instance();
    if(!global.isMain()) {
    	state.apply<size_t>(RunState::Keys::FRAME_COUNT, [](size_t& v){ return v++; });

		if(debugFlags() & DebugFlags::ONSCREEN_CONTEXTS) {
			swapContextBuffers();
		}
    }
	if (global.isMain()) {
		auto start = state.get<uint64_t>(RunState::Keys::START_TIME);
		auto now = get_epoch_nanos();
		auto diff = now - start;
		double diffSeconds = diff / 1000000000.0;

		if(state.get<double>(RunState::Keys::FPS) > 0 && diffSeconds > 1.0) {
			state.apply<uint64_t>(RunState::Keys::START_TIME, [diff](uint64_t& v) { return (v = v + (diff / 2.0)); } );
			state.apply<size_t>(RunState::Keys::FRAME_COUNT, [](size_t& v){ return (v = v * 0.5); });
		} else {
			double fps = state.get<double>(RunState::Keys::FPS);
			size_t cnt = state.get<size_t>(RunState::Keys::FRAME_COUNT);
			state.set<double>(RunState::Keys::FPS, (fps * 3.0 + (cnt / diffSeconds)) / 4.0);
		}

		if(getPrintFPS())
			std::cerr << "\rFPS:" << state.get<double>(RunState::Keys::FPS) << std::endl;
		{
			FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
			cv::Rect initial = get<cv::Rect>(Keys::INIT_VIEWPORT);
			initial.y = (fbCtx()->size().height - initial.height) + initial.y;
			fbCtx()->blitFrameBufferToFrameBuffer(initial, size(), 0, get<bool>(Keys::STRETCHING));
		}

		if(hasImguiCtx()) {
			GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
#if !defined(OPENCV_V4D_USE_ES3)
			GL_CHECK(glDrawBuffer(GL_BACK));
#endif
			imguiCtx()->render(getShowFPS());
		}
		TimeTracker::getInstance()->newCount();
		glfwSwapBuffers(fbCtx()->getGLFWWindow());
		state.set(RunState::Keys::DISPLAY_READY, true);
		GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		GL_CHECK(glViewport(0, 0, size().width, size().height));
		GL_CHECK(glClearColor(0,0,0,1));
		GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
		return !glfwWindowShouldClose(getGLFWWindow());
	} else {
		if(state.apply<bool>(RunState::Keys::DISPLAY_READY, [](bool& v){
			if(!v)
				return v;
			else {
				bool ret = v;
				v = !v;
				return ret;
			}
		})) {
			fbCtx()->copyToRootWindow();
		}
		if(debugFlags() & DebugFlags::ONSCREEN_CONTEXTS) {
			FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
			cv::Rect initial = get<cv::Rect>(Keys::INIT_VIEWPORT);
			initial.y = (fbCtx()->size().height - initial.height) + initial.y;
			fbCtx()->blitFrameBufferToFrameBuffer(initial, size(), 0, get<bool>(Keys::STRETCHING));
			glfwSwapBuffers(fbCtx()->getGLFWWindow());
		}
	}

    return true;
}

void V4D::setSequenceNumber(size_t seq) {
	seqNr_ = seq;
}

uint64_t V4D::getSequenceNumber() {
	//0 is an illegal sequence number
	CV_Assert(seqNr_ > 0);
	return seqNr_;
}

bool V4D::isClosed() {
    return fbCtx()->isClosed();
}

void V4D::close() {
    fbCtx()->close();
}

GLFWwindow* V4D::getGLFWWindow() const {
    return fbCtx()->getGLFWWindow();
}

void V4D::printSystemInfo() {
	std::cerr << "OpenGL: " << getGlInfo() << std::endl;
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL())
		std::cerr << "OpenCL Platforms: " << getClInfo() << endl;
#endif
}

int V4D::allocateFlags() {
	return allocateFlags_;
}

int V4D::configFlags() {
	return configFlags_;
}

int V4D::debugFlags() {
	return debugFlags_;
}

}
}
