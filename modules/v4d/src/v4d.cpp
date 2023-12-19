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

cv::Ptr<V4D> V4D::make(const cv::Size& size, const string& title, int allocFlags, int confFlags, int debFlags, int samples) {
    V4D* v4d = new V4D(size, cv::Size(), title, allocFlags, confFlags, debFlags, samples);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, int allocFlags, int confFlags, int debFlags, int samples) {
    V4D* v4d = new V4D(size, fbsize, title, allocFlags, confFlags, debFlags, samples);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const V4D& other, const string& title) {
    V4D* v4d = new V4D(other, title);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, int allocFlags, int confFlags, int debFlags, int samples) :
        initialSize_(size), allocateFlags_(allocFlags), configFlags_(confFlags), debugFlags_(debFlags), viewport_(0, 0, size.width, size.height), stretching_(true), samples_(samples) {
    self_ = cv::Ptr<V4D>(this);
    int fbFlags = FBConfigFlags::VSYNC
    		| (debugFlags() &  DebugFlags::DEBUG_GL_CONTEXT ? FBConfigFlags::DEBUG_GL_CONTEXT : 0)
			| (debugFlags() &  DebugFlags::ONSCREEN_CONTEXTS ? FBConfigFlags::ONSCREEN_CHILD_CONTEXTS : 0)
			| (configFlags() &  ConfigFlags::OFFSCREEN ? FBConfigFlags::OFFSCREEN : 0);
    mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, title, 3,
                2, samples, nullptr, nullptr, true, fbFlags);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);

    if(allocateFlags() & AllocateFlags::IMGUI)
        imguiContext_ = new detail::ImGuiContextImpl(mainFbContext_);

    setVisible(!(configFlags() & ConfigFlags::OFFSCREEN));
}

V4D::V4D(const V4D& other, const string& title) :
        initialSize_(other.initialSize_), allocateFlags_(other.allocateFlags_), configFlags_(other.configFlags_), debugFlags_(other.debugFlags_), viewport_(0, 0, other.fbSize().width, other.fbSize().height), stretching_(other.stretching_), samples_(other.samples_) {
	workerIdx_ = Global::on<size_t>(Global::WORKERS_INDEX, [](size_t& v){ return v++; });
    self_ = cv::Ptr<V4D>(this);
    int fbFlags = (configFlags() &  ConfigFlags::DISPLAY_MODE ? FBConfigFlags::DISPLAY_MODE : 0)
    		| (debugFlags() &  DebugFlags::DEBUG_GL_CONTEXT ? FBConfigFlags::DEBUG_GL_CONTEXT : 0)
			| (debugFlags() &  DebugFlags::ONSCREEN_CONTEXTS ? FBConfigFlags::ONSCREEN_CHILD_CONTEXTS : FBConfigFlags::OFFSCREEN);

    mainFbContext_ = new detail::FrameBufferContext(*this, other.fbSize(), title, 3,
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

const string V4D::getCurrentID() const {
	return currentID_;
}

cv::Ptr<V4D> V4D::setCurrentID(const string& id) {
	currentID_ = id;
	return self();
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

cv::Rect& V4D::viewport() {
    return viewport_;
}

cv::Rect V4D::getFramebufferViewport() {
	return fbCtx()->getViewport();
}

cv::Ptr<V4D> V4D::setFramebufferViewport(const cv::Rect& vp) {
	fbCtx()->setViewport(vp);
	return self();
}

float V4D::pixelRatioX() {
    return fbCtx()->pixelRatioX();
}

float V4D::pixelRatioY() {
    return fbCtx()->pixelRatioY();
}

const cv::Size& V4D::fbSize() const {
    return fbCtx()->size();
}

const cv::Size& V4D::initialSize() const {
    return initialSize_;
}

const cv::Size V4D::size() {
    return fbCtx()->getWindowSize();
}

void V4D::setSize(const cv::Size& sz) {
    fbCtx()->setWindowSize(sz);
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

cv::Ptr<V4D> V4D::setDisableIO(bool d) {
	disableIO_ = d;
	return self();
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

void V4D::setStretching(bool s) {
    stretching_ = s;
}

bool V4D::isStretching() {
    return stretching_;
}

cv::Ptr<Plan> V4D::plan() {
	return plan_;
}
void V4D::setFocused(bool f) {
    focused_ = f;
}

bool V4D::isFocused() {
    return focused_;
}

void V4D::swapContextBuffers() {
    for(int32_t i = -1; i < numGlCtx(); ++i) {
        FrameBufferContext::GLScope glScope(glCtx(i)->fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx(i)->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), glCtx(i)->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
        glfwSwapBuffers(glCtx(i)->fbCtx()->getGLFWWindow());
    }

    if(hasNvgCtx()) {
		FrameBufferContext::GLScope glScope(nvgCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
		nvgCtx()->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), nvgCtx()->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
		glfwSwapBuffers(nvgCtx()->fbCtx()->getGLFWWindow());
    }

    if(hasBgfxCtx()) {
		FrameBufferContext::GLScope glScope(bgfxCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
		bgfxCtx()->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), bgfxCtx()->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
		glfwSwapBuffers(bgfxCtx()->fbCtx()->getGLFWWindow());
    }

}

bool V4D::display() {
    if(!Global::is_main()) {
    	Global::on<size_t>(Global::FRAME_COUNT, [](size_t& v){ return v++; });

		if(debugFlags() & DebugFlags::ONSCREEN_CONTEXTS) {
			swapContextBuffers();
		}
    }
	if (Global::is_main()) {
		auto start = Global::get<uint64_t>(Global::START_TIME);
		auto now = get_epoch_nanos();
		auto diff = now - start;
		double diffSeconds = diff / 1000000000.0;

		if(Global::fps() > 0 && diffSeconds > 1.0) {
			Global::on<uint64_t>(Global::START_TIME, [diff](uint64_t& v) { return (v = v + (diff / 2.0)); } );
			Global::on<size_t>(Global::FRAME_COUNT, [](size_t& v){ return (v = v * 0.5); });
		} else {
			double fps = Global::fps();
			size_t cnt = Global::get<size_t>(Global::FRAME_COUNT);
			Global::set<double>(Global::FPS, (fps * 3.0 + (cnt / diffSeconds)) / 4.0);
		}

		if(getPrintFPS())
			cerr << "\rFPS:" << Global::fps() << endl;
		{
			FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
			fbCtx()->blitFrameBufferToFrameBuffer(viewport(), fbCtx()->getWindowSize(), 0, isStretching());
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
		Global::set(Global::DISPLAY_READY, true);
		GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		GL_CHECK(glViewport(0, 0, size().width, size().height));
		GL_CHECK(glClearColor(0,0,0,1));
		GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
		return !glfwWindowShouldClose(getGLFWWindow());
	} else {
		if(Global::on<bool>(Global::DISPLAY_READY, [](bool& v){
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
			fbCtx()->blitFrameBufferToFrameBuffer(viewport(), fbCtx()->getWindowSize(), 0, isStretching());
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
	cerr << "OpenGL: " << getGlInfo() << endl;
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL())
		cerr << "OpenCL Platforms: " << getClInfo() << endl;
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

cv::Ptr<V4D> V4D::self() {
       return self_;
}


}
}
