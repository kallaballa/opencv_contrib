// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_


#include "source.hpp"
#include "sink.hpp"
#include "util.hpp"
#include "nvg.hpp"
#include "detail/context.hpp"
#include "detail/transaction.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/imguicontext.hpp"
#include "detail/timetracker.hpp"
#include "detail/glcontext.hpp"
#include "detail/extcontext.hpp"
#include "detail/sourcecontext.hpp"
#include "detail/sinkcontext.hpp"
#include "detail/bgfxcontext.hpp"
#include "detail/resequence.hpp"
#include "threadsafeanymap.hpp"
#define EVENT_API_EXPORT CV_EXPORTS
#include "events.hpp"

#include <shared_mutex>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <barrier>
#include <type_traits>
#include <sys/resource.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

/*!
 * OpenCV namespace
 */
namespace cv {
/*!
 * V4D namespace
 */
namespace v4d {

using namespace std::chrono_literals;
using namespace cv::utils::logging;

const LogTag cf_tag("Flow", LogLevel::LOG_LEVEL_INFO);
const LogTag v4d_tag("V4D", LogLevel::LOG_LEVEL_INFO);
const LogTag mon_tag("Monitor", LogLevel::LOG_LEVEL_INFO);

namespace event {
	typedef Mouse_<cv::Point> Mouse;
	typedef Window_<cv::Point> Window;
	inline std::vector<std::shared_ptr<Mouse>> fetch(const Mouse::Type& t){
		return fetch<Mouse>(t);
	}

	inline std::vector<std::shared_ptr<Mouse>> fetch(const Mouse::Type& t, const Mouse::Button& b){
		return fetch<Mouse>(t ,b);
	}

	inline std::vector<std::shared_ptr<Window>> fetch(const Window::Type& t){
		return fetch<Window>(t);
	}

	inline bool consume(const Mouse::Type& t){
		return consume<Mouse>(t);
	}

	inline bool consume(const Mouse::Type& t, const Mouse::Button& b){
		return consume<Mouse>(t ,b);
	}

	inline bool consume(const Window::Type& t){
		return consume<Window>(t);
	}

}
struct AllocateFlags {
	enum Enum {
		NONE = 0,
		NANOVG = 1,
		IMGUI = 2,
		BGFX = 4,
		DEFAULT = NONE
	};
};

struct ConfigFlags {
	enum Enum {
		DEFAULT = 0,
		OFFSCREEN = 1,
		DISPLAY_MODE = 2,
	};
};

struct DebugFlags {
	enum Enum {
		DEFAULT = 0,
		ONSCREEN_CONTEXTS = 1,
		PRINT_CONTROL_FLOW = 2,
		DEBUG_GL_CONTEXT = 4,
		PRINT_LOCK_CONTENTION = 8,
		MONITOR_RUNTIME_PROPERTIES = 16,
		LOWER_WORKER_PRIORITY = 32,
		DONT_PAUSE_LOG = 64,
	};
};

using namespace cv::v4d::detail;

/*!
 * Private namespace
 */
namespace detail {

template <typename T> using static_not = std::integral_constant<bool, !T::value>;

template<typename T> std::string int_to_hex( T i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(sizeof(T) * 2)
         << std::hex << i;
  return stream.str();
}

template<typename Tlamba> std::string lambda_ptr_hex(Tlamba&& l) {
    return int_to_hex((size_t)Lambda::ptr(l));
}

static std::size_t map_index(const std::thread::id id) {
    static std::size_t nextindex = 0;
    static std::mutex my_mutex;
    static std::unordered_map<std::thread::id, std::size_t> ids;
    std::lock_guard<std::mutex> lock(my_mutex);
    auto iter = ids.find(id);
    if(iter == ids.end())
        return ids[id] = nextindex++;
    return iter->second;
}

}
class Plan;
class CV_EXPORTS V4D {
    friend class detail::FrameBufferContext;
    friend class detail::SourceContext;
    friend class detail::SinkContext;
    friend class detail::NanoVGContext;
    friend class detail::ImGuiContextImpl;
    friend class detail::PlainContext;
    friend class detail::GLContext;
    friend class detail::ExtContext;
    friend class detail::BgfxContext;
    friend class Source;
    friend class Sink;
    friend class Plan;
public:
    struct Keys {
    	enum Enum {
    		INIT_VIEWPORT,
    		VIEWPORT,
			WINDOW_SIZE,
			FB_SIZE,
			STRETCHING,
			CLEAR_COLOR,
			NAMESPACE,
			FULLSCREEN,
			DISABLE_VIDEO_IO
    	};
    };
private:
    CV_EXPORTS static thread_local std::mutex instance_mtx_;
    CV_EXPORTS static thread_local cv::Ptr<V4D> instance_;
    CV_EXPORTS static ThreadSafeAnyMap<Keys::Enum> properties_;

    int32_t workerIdx_ = -1;

    int allocateFlags_;
    int configFlags_;
    int debugFlags_;

    int samples_;
    cv::Ptr<FrameBufferContext> mainFbContext_ = nullptr;
    cv::Ptr<SourceContext> sourceContext_ = nullptr;
    cv::Ptr<SinkContext> sinkContext_ = nullptr;
    cv::Ptr<NanoVGContext> nvgContext_ = nullptr;
    cv::Ptr<BgfxContext> bgfxContext_ = nullptr;
    cv::Ptr<ImGuiContextImpl> imguiContext_ = nullptr;
    cv::Ptr<PlainContext> plainContext_ = nullptr;
    std::mutex glCtxMtx_;
    std::map<int32_t,cv::Ptr<GLContext>> glContexts_;
    std::map<int32_t,cv::Ptr<ExtContext>> extContexts_;
    bool closed_ = false;
    cv::Ptr<Source> source_;
    cv::Ptr<Sink> sink_;
    uint64_t seqNr_ = 0;
    bool showFPS_ = true;
    bool printFPS_ = false;
    bool showTracking_ = true;
    std::string currentID_;
public:
    CV_EXPORTS static cv::Ptr<V4D> instance() {
    	std::lock_guard guard(instance_mtx_);
    	if(!instance_)
    		CV_Error(cv::Error::StsAssert, "Runtime not initialized. You have to call the static ```init``` function of the runtime first!");
    	return instance_;
    }

    template<typename Tval>
	void set(Keys::Enum key, const Tval& val, bool fire = true) {
    	if(instance()->debugFlags() & DebugFlags::MONITOR_RUNTIME_PROPERTIES) {
    		stringstream ss;
    		ss << demangle(typeid(decltype(key)).name()) << " = " << val << " (fire: " << fire << ")";
    		CV_LOG_INFO(&mon_tag, ss.str());
    	}
		properties_.set(key, val, fire);
	}

    template<typename Tval>
	const auto& get(Keys::Enum key) const {
		return properties_.get<Tval>(key);
	}

    /*!
     * Creates a V4D object which is the central object to perform visualizations with.
     * @param initialSize The initial size of the heavy-weight window.
     * @param frameBufferSize The initial size of the framebuffer backing the window (needs to be equal or greate then initial size).
     * @param offscreen Don't create a window and rather render offscreen.
     * @param title The window title.
     * @param major The OpenGL major version to request.
     * @param minor The OpenGL minor version to request.
     * @param compat Request a compatibility context.
     * @param samples MSAA samples.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS static cv::Ptr<V4D> init(const cv::Rect& viewport, const string& title, int allocateFlags = AllocateFlags::DEFAULT, int configFlags = ConfigFlags::DEFAULT, int debugFlags = DebugFlags::DEFAULT, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> init(const cv::Rect& viewport, const cv::Size& fbsize, const string& title, int allocateFlags = AllocateFlags::DEFAULT, int configFlags = ConfigFlags::DEFAULT, int debugFlags = DebugFlags::DEFAULT, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> init(const V4D& v4d, const string& title);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();
    CV_EXPORTS const int32_t& workerIndex() const;
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS std::string title() const;


    /*!
     * Copy the framebuffer contents to an OutputArray.
     * @param arr The array to copy to.
     */
    CV_EXPORTS void copyTo(cv::UMat& arr);
    /*!
     * Copy the InputArray contents to the framebuffer.
     * @param arr The array to copy.
     */
    CV_EXPORTS void copyFrom(const cv::UMat& arr);

    /*!
     * Set the current #cv::viz::Source object. Usually created using #makeCaptureSource().
     * @param src A #cv::viz::Source object.
     */
    CV_EXPORTS void setSource(cv::Ptr<Source> src);
    CV_EXPORTS cv::Ptr<Source> getSource();
    CV_EXPORTS bool hasSource() const;

    /*!
     * Set the current #cv::viz::Sink object. Usually created using #Sink::make().
     * @param sink A #cv::viz::Sink object.
     */
    CV_EXPORTS void setSink(cv::Ptr<Sink> sink);
    CV_EXPORTS cv::Ptr<Sink> getSink();
    CV_EXPORTS bool hasSink() const;
    /*!
     * Get the window position.
     * @return The window position.
     */
    CV_EXPORTS cv::Vec2f position();

    /*!
     * Get the pixel ratio of the display x-axis.
     * @return The pixel ratio of the display x-axis.
     */
    CV_EXPORTS float pixelRatioX();
    /*!
     * Get the pixel ratio of the display y-axis.
     * @return The pixel ratio of the display y-axis.
     */
    CV_EXPORTS float pixelRatioY();
    /*!
     * Get the window size.
     * @return The window size.
     */
    CV_EXPORTS const cv::Size& size();
    /*!
     * Get the frambuffer size.
     * @return The framebuffer size.
     */

    CV_EXPORTS bool getShowFPS();
    CV_EXPORTS void setShowFPS(bool s);
    CV_EXPORTS bool getPrintFPS();
    CV_EXPORTS void setPrintFPS(bool p);
    CV_EXPORTS bool getShowTracking();
    CV_EXPORTS void setShowTracking(bool st);
    CV_EXPORTS void printSystemInfo();
    CV_EXPORTS int allocateFlags();
    CV_EXPORTS int configFlags();
    CV_EXPORTS int debugFlags();

    static void run(cv::Ptr<V4D> runtime, std::function<void()> runGraph) {
		//the first sequence number is 1!
		static Resequence reseq(1);
    	static std::binary_semaphore frame_sync_render(0);
		static std::binary_semaphore frame_sync_sema_swap(0);
		Global& global = Global::instance();
		RunState& state = RunState::instance();

		try {
			if(global.isMain()) {
				CV_LOG_INFO(&v4d_tag, "Display thread started.");
				while(keepRunning()) {
					bool result = true;
					TimeTracker::getInstance()->execute("display", [&result, runtime](){
					if(runtime->configFlags() & ConfigFlags::DISPLAY_MODE) {
						if(!runtime->display()) {
							frame_sync_render.release();
							result = false;
						}
						frame_sync_render.release();
						//refresh-rate depends on swap interval (1) for sync
						frame_sync_sema_swap.acquire();
					} else {
						if(!runtime->display()) {
							result = false;
						}
					}
					});
					if(!result)
						break;
				}
			} else {
				while(keepRunning()) {
					bool result = true;
					TimeTracker::getInstance()->execute("worker", [&result, &state, runtime, runGraph](){
						event::poll();
						if(!runtime->hasSource() || (runtime->hasSource() && !runtime->getSource()->isOpen())) {
							state.apply<size_t>(RunState::Keys::RUN_COUNT, [runtime](size_t& s) {
								runtime->setSequenceNumber(++s);
								return s;
							});
						}

						if(runtime->configFlags() & ConfigFlags::DISPLAY_MODE) {
							frame_sync_sema_swap.release();
							runGraph();
							size_t seq = runtime->getSequenceNumber();
							reseq.waitFor(seq, [](uint64_t s) {
								frame_sync_render.acquire();
							});

							if(!runtime->display()) {
								frame_sync_sema_swap.release();
								result = false;
							}
						} else {
							runGraph();
							reseq.waitFor(runtime->getSequenceNumber(), [&result, runtime](uint64_t s) {
								CV_UNUSED(s);
								result = runtime->display();
							});
						}
					});
					if(!result)
						break;
				}
			}
		} catch(std::runtime_error& ex) {
			CV_LOG_WARNING(&v4d_tag, "Pipeline terminated: " << ex.what());
		} catch(std::exception& ex) {
			CV_LOG_WARNING(&v4d_tag, "Pipeline terminated: " << ex.what());
		} catch(...) {
			CV_LOG_WARNING(&v4d_tag, "Pipeline terminated with unknown error.");
		}
		requestFinish();
		reseq.finish();
		if(runtime->configFlags() & ConfigFlags::DISPLAY_MODE) {
			if(global.isMain()) {
				for(size_t i = 0; i < state.get<size_t>(RunState::Keys::WORKERS_STARTED); ++i)
					frame_sync_render.release();
			} else {
				frame_sync_sema_swap.release();
			}
    	}
    }
    void setSequenceNumber(size_t seq);
    uint64_t getSequenceNumber();
private:
    V4D(const V4D& v4d, const string& title);
    V4D(const cv::Rect& size, cv::Size fbsize,
            const string& title, int allocFlags, int confFlags, int debFlags, int samples);

    void swapContextBuffers();
    bool display();
protected:
	template<bool Tread, typename Tval>
	void create(Keys::Enum key, const Tval& val, const std::function<void(const Tval& val)>& cb = std::function<void(const Tval& val)>()) {
		properties_.create<Tread>(key, val, cb);
	}

    cv::Ptr<FrameBufferContext> fbCtx() const;
    cv::Ptr<SourceContext> sourceCtx();
    cv::Ptr<SinkContext> sinkCtx();
    cv::Ptr<NanoVGContext> nvgCtx();
    cv::Ptr<BgfxContext> bgfxCtx();
    cv::Ptr<PlainContext> plainCtx();
    cv::Ptr<ImGuiContextImpl> imguiCtx();
    cv::Ptr<GLContext> glCtx(int32_t idx = 0);
    cv::Ptr<ExtContext> extCtx(int32_t idx = 0);

    bool hasFbCtx();
    bool hasSourceCtx();
    bool hasSinkCtx();
    bool hasNvgCtx();
    bool hasBgfxCtx();
    bool hasPlainCtx();
    bool hasImguiCtx();
    bool hasGlCtx(uint32_t idx = 0);
    bool hasExtCtx(uint32_t idx = 0);
    int32_t numGlCtx();
    int32_t numExtCtx();

    GLFWwindow* getGLFWWindow() const;
    bool isFocused();
    void setFocused(bool f);
};
class Plan {
	friend class V4D;
    friend class detail::FrameBufferContext;
    friend class detail::EdgeBase;
    struct BranchState {
		string branchID_;
    	bool isEnabled_ = true;
    	bool isOnce_ = false;
    	bool isSingle_ = false;
    	bool condition_ = false;
    	bool isLocked_ = false;
    };

	cv::Ptr<V4D> runtime_ = V4D::instance();
	std::string parent_;
    cv::UMat captureFrame_;
    cv::UMat writerFrame_;
    size_t parentOffset_ = 0;
    size_t parentActualTypeSize_ = 0;
    size_t actualTypeSize_ = 0;
    cv::Ptr<Plan> self_;
    std::vector<std::tuple<std::string,bool,size_t>> accesses_;
    std::map<std::string, cv::Ptr<Transaction>> transactions_;
    std::vector<cv::Ptr<Node>> nodes_;
    std::deque<BranchState> branchStateStack_;
    std::deque<std::pair<string, BranchType::Enum>> branchStack_;

	template<typename Tedge>
    void emit_access(const string& context, Tedge tp) {
//    	cout << "access: " << std::this_thread::get_id() << " " << context << string(read ? " <- " : " -> ") << demangle(typeid(std::remove_const_t<T>).name()) << "(" << (long)tp << ") " << endl;
    	accesses_.push_back(std::make_tuple(context, Tedge::read_t::value, tp.id()));
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(std::function<cv::Ptr<V4DContext>()> ctxCb, string txID, Tfn fn, Args ...args) {
		auto tx = make_transaction(fn, args...);
		tx->setContextCallback(ctxCb);
		tx->setBranchType(BranchType::NONE);
		transactions_.insert({txID, tx});
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(BranchType::Enum btype, std::function<cv::Ptr<V4DContext>()> ctxCb, string txID, Tfn fn, Args ...args) {
		auto tx = make_transaction(fn, args...);
		tx->setContextCallback(ctxCb);
		tx->setBranchType(btype);
		transactions_.insert({txID, tx});
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(cv::Ptr<V4DContext> ctx, const string& txID, Tfn fn, Args ...args) {
    	this->add_transaction([ctx](){ return ctx; }, txID, fn, args...);
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(BranchType::Enum btype, cv::Ptr<V4DContext> ctx, const string& txID, Tfn fn, Args ...args) {
    	this->add_transaction(btype, [ctx](){ return ctx; }, txID, fn, args...);
    }

    template <typename T>
    struct ReturnType {
    	using type = typename std::result_of<T>::type;
    };

    template <typename Return, typename Object>
    struct ReturnType<Return Object::*>
    {
        using type = Return;
    };

    template <typename Return, typename Object, typename... Args>
    struct ReturnType<Return (Object::*)(Args...)>
    {
        using type = Return;
    };

    template <typename Return, typename... Args>
    struct ReturnType<Return (*)(Args...)>
    {
        using type = Return;
    };

    template <typename TReturn = std::false_type, typename Tfn, typename ... Args>
    auto wrap_callable(Tfn fn, Args ... args) {
//    	static_assert(std::is_invocable_v<Tfn>, "Error: You passed a non-invocable as function argument.");
    	if constexpr(std::is_same<TReturn, std::false_type>::value) {
    		return std::function<typename ReturnType<decltype(fn)>::type(typename Args::ref_t...)>(fn);
    	} else {
    		return std::function<TReturn(typename Args::ref_t...)>(fn);
    	}
    }

	template<typename T>
	detail::Edge<T, false, true> R_I(T& t) {
		return detail::Edge<T, false, true>::make(*this, t, false);
	}

	template<typename T>
	detail::Edge<T, true, true> R_C_I(T& t) {
		return detail::Edge<T, true, true>::make(*this, t, false);
	}

	template<typename T>
	detail::Edge<T, false, false> RW_I(T& t) {
		return detail::Edge<T, false, false>::make(*this, t, false);
	}

//	template<typename T>
//	detail::Edge<T, true, false> RW_C_I(T& t) {
//		return detail::Edge<T, true, false>::make(*this, t, false);
//	}

	template<typename T>
	detail::Edge<cv::Ptr<T>, false, true, false, T> VAL_I(T t) {
		cv::Ptr<T> ptr = new T(t);
		return detail::Edge<decltype(ptr), false, true, false, T>::make(*this, ptr, false);
	}

    template<bool Tconst, typename T>
    auto makeInternalEdge(T& val) {
		if constexpr(Tconst) {
			return R_I(val);
		} else {
			return RW_I(val);
		}
    }

    template<typename T>
    void setActualTypeSize() {
    	actualTypeSize_ = sizeof(T);
    }

    template<typename T>
    void setParentActualTypeSize() {
    	parentActualTypeSize_ = sizeof(T);
    }

    void setParentOffset(size_t offset) {
    	parentOffset_ = offset;
    }

	size_t getActualTypeSize() {
    	return actualTypeSize_;
    }

	size_t getParentActualTypeSize() {
    	return parentActualTypeSize_;
    }

	size_t getParentOffset() {
    	return parentOffset_;
    }

	void findNode(const string& name, cv::Ptr<Node>& found) {
    	CV_Assert(!name.empty());
    	if(nodes_.empty())
    		return;

    	if(nodes_.back()->name_ == name)
    		found = nodes_.back();
    }

    void makeGraph() {
 //    	cout << std::this_thread::get_id() << " ### MAKE PLAN ### " << endl;
     	for(const auto& t : accesses_) {
     		const string& name = std::get<0>(t);

     		const bool& read = std::get<1>(t);
     		const size_t& dep = std::get<2>(t);
     		cv::Ptr<Node> n;
     		findNode(name, n);

     		if(!n) {
     			n = new Node();
     			n->name_ = name;
     			n->tx_ = transactions_[name];
     			CV_Assert(!n->name_.empty());
     			CV_Assert(n->tx_);
     			nodes_.push_back(n);
 //        		cout << "make: " << std::this_thread::get_id() << " " << n->name_ << endl;
     		}


     		if(read) {
     			n->read_deps_.insert(dep);
     		} else {
     			n->write_deps_.insert(dep);
     		}
     	}
     }

    //shortened name because it shows up in the log output
    void pf(const size_t& depth, const BranchState& current, const cv::Ptr<Node> n) {
    	if(DebugFlags::PRINT_CONTROL_FLOW & runtime_->debugFlags()) {
    		std::stringstream indent;
			indent << "|";
    		for(size_t i = 0; i < depth; ++i) {
				indent << "  ";
			}

    		std::stringstream ss;

			//delete addresses from name
			string name = n->name_;
			size_t offset = name.find_first_of(':', 0) + 1;

			size_t open = 0;
			size_t close = 0;
			while(true) {
				open = name.find_first_of('[', offset);
				if(open == string::npos)
					break;
				close = name.find_first_of(']', open);
				if(close == string::npos)
					break;
				CV_Assert(name.size() > close + 1);
				name.erase(open, close + 1 - open);
			}

			open = 0;
			close = 0;
			while(true) {
				open = name.find_first_of('(', offset);
				if(open == string::npos)
					break;
				close = name.find_first_of(')', open);
				if(close == string::npos)
					break;
				CV_Assert(name.size() > close + 1);
				name.erase(open, close + 1 - open);
			}

			ss << indent.str() << name;
			const string formattedName = ss.str();
			ss.str("");
			ss << indent.str() << "-> (enabled: " << current.isEnabled_ << ") "
					<< "(once: " << current.isOnce_ << ") "
					<< "(single: " << current.isSingle_ << ") "
					<< "(branch lock: " << current.isLocked_ << ") "
					<< "(shared lock: " << n->tx_->hasLockies() << ")";
			const string formattedInfo = ss.str();
			static std::mutex printMtx;
			std::lock_guard guard(printMtx);
			CV_LOG_INFO(&cf_tag, formattedName);
			CV_LOG_INFO(&cf_tag, formattedInfo);
    	}
    }

    void runGraph() {
		BranchType::Enum btype;
    	BranchState currentState;
    	Global& global = Global::instance();
    	bool countLockContention = DebugFlags::PRINT_LOCK_CONTENTION & runtime_->debugFlags();
    	try {
			for (auto& n : nodes_) {
				btype = n->tx_->getBranchType();
				bool isBranch = n->name_.substr(0, 6) == "branch";
				bool isElse = n->name_.substr(0,6) == "[else]";
				bool isEnd = n->name_.substr(0,5) == "[end]";
				bool isElseIf = n->name_.substr(0,8) == "[elseif]";
				if(btype != BranchType::NONE) {
					CV_Assert((((isBranch != isElse) != isEnd) != isElseIf));
					if(isBranch) {
						if(!branchStateStack_.empty())
							currentState = branchStateStack_.front();
						else
							currentState = BranchState();
						currentState.branchID_ = n->name_;

						if(currentState.isEnabled_) {
							currentState.isOnce_ = ((btype == BranchType::ONCE) || (btype == BranchType::PARALLEL_ONCE));
							currentState.isSingle_ = ((btype == BranchType::ONCE) || (btype == BranchType::SINGLE));
						} else {
							currentState.isOnce_ = false;
							currentState.isSingle_ = false;
							currentState.isEnabled_ = false;
						}

						if(currentState.isEnabled_) {
							if(currentState.isOnce_) {
								if((btype == BranchType::ONCE)) {
									currentState.condition_ = global.once(n->name_) && n->tx_->performPredicate(countLockContention);
								} else if((btype == BranchType::PARALLEL_ONCE)) {
									currentState.condition_ = !n->tx_->ran() && n->tx_->performPredicate(countLockContention);;
								} else {
									CV_Assert(false);
								}
							} else {
								currentState.condition_ = n->tx_->performPredicate(countLockContention);
							}

							currentState.isEnabled_ = currentState.isEnabled_ && currentState.condition_;

							if(currentState.isEnabled_ && currentState.isSingle_) {
								CV_Assert(btype != BranchType::PARALLEL);

								if(global.lockNode(currentState.branchID_)) {
	//								cerr << "lock branch" << endl;
								}
								currentState.isLocked_ = true;
							}
						}
						branchStateStack_.push_front(currentState);
						pf(branchStateStack_.size(), currentState, n);
					} else if(isElse) {
						if(branchStateStack_.empty())
							continue;
						currentState = branchStateStack_.front();
						currentState.isEnabled_ = !currentState.condition_;
						currentState.isOnce_ = false;
						currentState.condition_ = !currentState.condition_;
						currentState.isSingle_ = false;

						if(currentState.isLocked_) {
							if(global.tryUnlockNode(currentState.branchID_)) {
//								cerr << "unlock else" << endl;
							}
						}

						currentState.isLocked_ = false;
						pf(branchStateStack_.size(), currentState, n);
						branchStateStack_.pop_front();
						branchStateStack_.push_front(currentState);
					} else if(isEnd) {
						if(branchStateStack_.empty())
							continue;

						currentState = branchStateStack_.front();
						if(global.tryUnlockNode(currentState.branchID_)) {
//							cerr << "unlock end" << endl;
						}
						pf(branchStateStack_.size(), currentState, n);
						branchStateStack_.pop_front();
					} else {
						CV_Assert(false);
					}
				} else {
					CV_Assert(!n->tx_->isPredicate());
					currentState = !branchStateStack_.empty() ? branchStateStack_.front() : BranchState();
					const bool disableIO = runtime_->get<bool>(V4D::Keys::DISABLE_VIDEO_IO);
					if(currentState.isEnabled_) {
						auto lock = global.tryGetNodeLock(currentState.branchID_);
						if(lock)
						{
							std::lock_guard<std::mutex> guard(*lock.get());
							auto ctx = n->tx_->getContextCallback()();
							auto viewport = runtime_->get<cv::Rect>(V4D::Keys::VIEWPORT);
							int res = ctx->execute(viewport, [countLockContention, n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [countLockContention, n,currentState](){
//									cerr << "locked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform(countLockContention);
								});
							});
							if(res > 0) {
								if(!disableIO && std::dynamic_pointer_cast<SourceContext>(ctx)) {
									runtime_->setSequenceNumber(res);
								}
							} else {
								CV_LOG_WARNING(&v4d_tag, "Context failed while: " + n->name_);
							}
						} else {
							auto ctx = n->tx_->getContextCallback()();
							auto viewport = runtime_->get<cv::Rect>(V4D::Keys::VIEWPORT);
							int res = ctx->execute(viewport, [countLockContention, n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [countLockContention, n,currentState](){
//									cerr << "unlocked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform(countLockContention);
								});
							});
							if(res > 0) {
								if(!disableIO && dynamic_pointer_cast<SourceContext>(ctx)) {
									runtime_->setSequenceNumber(res);
								}
							} else {
								CV_LOG_WARNING(&v4d_tag, "Context failed while: " + n->name_);
							}
						}
					}
					pf(branchStateStack_.size() +1 , currentState, n);
					currentState = BranchState();
				}
			}

			size_t lockCnt = global.countNodeLocks();
//			cerr << "STATE STACK: " << branchStateStack_.size() << endl;
//			cerr << "LOCK STACK: " << lockCnt << endl;
			CV_Assert(branchStateStack_.empty());
			CV_Assert(lockCnt == 0);
			//FIXME unlock all on exception?
    	} catch(std::runtime_error& ex) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(global.tryUnlockNode(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw ex;
		} catch(std::exception& ex) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(global.tryUnlockNode(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw ex;
		} catch(...) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(global.tryUnlockNode(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw std::runtime_error("Unkown error.");
		}
	}

	void clearGraph() {
		accesses_.clear();
		branchStateStack_.clear();
		branchStack_.clear();
		transactions_.clear();
		nodes_.clear();
	}

	size_t id_ = Global::instance().apply<size_t>(Global::Keys::PLAN_CNT, [](size_t& v){
		++v;
		return v;
	});

	template<typename Tinstance>
    cv::Ptr<Tinstance> self() {
		return self_.dynamicCast<Tinstance>();
	}

    template<typename Tplan, typename Tparent, typename ... Args>
	static cv::Ptr<Tplan> makeSubPlan(Tparent* parent, Args&& ... args) {
    	cv::Ptr<Tplan> plan = std::make_shared<Tplan>(std::forward<Args>(args)...);
    	plan->self_ = plan;
    	plan->setParentID(parent->space());
    	plan->setParentOffset(reinterpret_cast<size_t>(parent));
    	plan->template setParentActualTypeSize<Tparent>();
    	plan->template setActualTypeSize<Tplan>();
		plan->runtime_->set(V4D::Keys::NAMESPACE, plan->space());
		return plan;
    }

    template<typename Tfn, typename ... Args>
    const string make_id(string id, const string& name, Tfn fn, Args ... args) {
    	stringstream ss;
    	if(!id.empty())
    		id = "::" + id;

    	if constexpr(std::is_pointer<Tfn>::value) {
    		ss << name << id << " [" << detail::int_to_hex(reinterpret_cast<size_t>(fn)) << "] ";
    	} else {
    		ss << name << id << " [" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << "] ";
    	}

    	((ss << demangle(typeid(typename std::remove_reference_t<decltype(args)>::ref_t).name()) << "(" << int_to_hex(args.id()) << ") "), ...);
    	ss << "- " <<  map_index(std::this_thread::get_id());
    	while(transactions_.find(ss.str()) != transactions_.end()) {
    				ss << '+';
    	}
    	return ss.str();
    }
public:
	template<typename T>
	struct Property : detail::Edge<const T, false, true, true> {
		using parent_t = detail::Edge<const T, false, true, true>;
		Property(Plan& plan, const T& val) : parent_t(parent_t::make(plan, val, false)) {
			Global::instance().registerShared<decltype(val),false>(val);
		}
	};

	//predefined branch predicates
	constexpr static auto always_ = []() { return true; };
	constexpr static auto isTrue_ = [](const bool& b) { return b; };
	constexpr static auto isFalse_ = [](const bool& b) { return !b; };
	constexpr static auto and_ = [](const bool& a, const bool& b) { return a && b; };
	constexpr static auto or_ = [](const bool& a, const bool& b) { return a || b; };

	virtual ~Plan() { self_ = nullptr; };
	virtual void gui() { };
	virtual void setup() { };
	virtual void infer() = 0;
	virtual void teardown() { };

	virtual std::string space() {
		if(!parent_.empty()) {
			return parent_ + "-" + name();
		} else
			return name();
	}

	virtual std::string name() {
		return detail::demangle(typeid(*this).name()) + std::to_string(id_);
	}

	virtual void setParentID(const string& parent) {
		parent_  = parent;
	}

	virtual std::string getParentID() {
		return parent_;
	}

    template <typename Tfn, typename ... Args>
    typename std::enable_if<is_callable<Tfn>::value, cv::Ptr<Plan>>::type
    gl(Tfn fn, Args ... args) {
    	auto wrap = wrap_callable<void>(fn, args...);
        const string id = make_id(this->space(), "gl-1", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->glCtx(-1), id, wrap, args...);
		return self<Plan>();
    }

    std::map<size_t, size_t> indexPointerMap_;
    template <typename Tedge, typename Tfn, typename ... Args>
    typename std::enable_if<!is_callable<Tedge>::value, cv::Ptr<Plan>>::type
	gl(Tedge indexEdge, Tfn fn, Args ... args) {
        auto wrap = wrap_callable<void>(fn, indexEdge, args...);
        const string id = make_id(this->space(), "gl-" + int_to_hex(indexEdge.ptr()), fn, args...);
        emit_access(id, R_I(*this));
        emit_access(id, indexEdge);
        (emit_access(id, args ),...);
        std::function<void((const int32_t&,typename Args::ref_t...))> functor(wrap);
		add_transaction([this, indexEdge](){
			Tedge copy = indexEdge;
			return runtime_->glCtx(copy.ref());},id, functor, indexEdge, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> ext(Tfn fn, Args ... args) {
    	auto wrap = wrap_callable<void>(fn, args...);
        const string id = make_id(this->space(), "ext", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->extCtx(-1), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tedge, typename Tfn, typename ... Args>
    cv::Ptr<Plan> ext(Tedge indexEdge, Tfn fn, Args ... args) {
        auto wrap = wrap_callable<void>(fn, args...);
        const string id = make_id(this->space(), "ext" + int_to_hex(indexEdge.ptr()), fn, args...);
        emit_access(id, R_I(*this));
        emit_access(id, indexEdge);
        (emit_access(id, args ),...);
        std::function<void((const int32_t&,typename Args::ref_t...))> functor(fn);
		add_transaction([this, indexEdge](){ return runtime_->extCtx(indexEdge.ref());},id, wrap, indexEdge, args...);
		return self<Plan>();
    }

    template <typename Tfn>
    cv::Ptr<Plan> branch(Tfn fn) {
        auto wrap = wrap_callable<bool>(fn);
        const string id = make_id(this->space(), "branch", fn);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R_I(*this));
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(Tfn fn, Args ... args) {
        auto wrap = wrap_callable<bool>(fn, args...);
        const string id = make_id(this->space(), "branch", fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(int workerIdx, Tfn fn, Args ... args) {
        auto wrapInner = wrap_callable<bool>(fn, args...);
        const string id = make_id(this->space(), "branch-pin" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		std::function<bool((typename Args::ref_t...))> wrap = [this, workerIdx, wrapInner](Args ... args){
			return runtime_->workerIndex() == workerIdx && wrapInner(args...);
		};
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(BranchType::Enum type, Tfn fn, Args ... args) {
        auto wrap = wrap_callable<bool>(fn, args...);
        const string id = make_id(this->space(), "branch-type" + std::to_string((int)type), fn, args...);
        branchStack_.push_front({id, type});
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(type, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(BranchType::Enum type, int workerIdx, Tfn fn, Args ... args) {
        auto wrapInner = wrap_callable<bool>(fn, args...);
        const string id = make_id(this->space(), "branch-type-pin" + std::to_string((int)type) + "-" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, type});
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		std::function<bool((typename Args::ref_t...))> wrap = [this, workerIdx, wrapInner](Args ... args){
			return runtime_->workerIndex() == workerIdx && wrapInner(args...);
		};

		add_transaction(type, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

/*    template <typename Tfn>
    cv::Ptr<Plan> elseIfBranch(Tfn fn) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
    	std::function functor = fn;
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, functor, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(int workerIdx, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
    	(emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
		std::function<bool(typename Args::ref_t...)> wrap = [this, workerIdx, functor](Args ... args){
			return runtime_->workerIndex() == workerIdx && functor(args...);
		};
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn>
    cv::Ptr<Plan> elseIfBranch(BranchType::Enum type, Tfn fn) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
    	std::function functor = fn;
		add_transaction(type, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(BranchType::Enum type, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
    	(emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
		add_transaction(type, runtime_->plainCtx(), id, functor, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(BranchType::Enum type, int workerIdx, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<void>(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
        std::function<bool(typename Args::ref_t...)> wrap = [this, workerIdx, functor](Args ... args){
			return runtime_->workerIndex() == workerIdx && functor(args...);
		};

		add_transaction(type, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }
*/
	cv::Ptr<Plan> endBranch() {
    	auto current = branchStack_.front();
    	branchStack_.pop_front();
        string id = "[end]" + current.first;
        emit_access(id, R_I(*this));
        std::function functor = [](){ return true; };
		add_transaction(current.second, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    cv::Ptr<Plan> elseBranch() {
    	auto current = branchStack_.front();
    	string id = "[else]" + current.first;
    	emit_access(id, R_I(*this));
		std::function functor = [](){ return true; };
		add_transaction(current.second, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> fb(Tfn fn, Args ... args) {
		using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;
		static_assert((std::is_same<Tfb, cv::UMat>::value || std::is_same<Tfb, const cv::UMat>::value) || !"The first argument must be eiter of type 'cv::UMat&' or 'const cv::UMat&'");
		auto fbEdge = makeInternalEdge<std::is_const<Tfb>::value>(runtime_->fbCtx()->fb());
    	auto wrap = wrap_callable<void>(fn, fbEdge, args...);
        const string id = make_id(this->space(), "fb", fn, args...);
		emit_access(id, R_I(*this));
		(emit_access(id, args ),...);

		std::function<void((
				typename decltype(fbEdge)::ref_t,
				typename Args::ref_t...))> functor(wrap);
		add_transaction(runtime_->fbCtx(),id, functor, fbEdge, args...);
		return self<Plan>();
    }

    cv::Ptr<Plan> clear() {
    	gl([](const cv::Scalar& bgra){
    		const float& b = bgra[0] / 255.0f;
		    const float& g = bgra[1] / 255.0f;
		    const float& r = bgra[2] / 255.0f;
		    const float& a = bgra[3] / 255.0f;
		    GL_CHECK(glClearColor(r, g, b, a));
		    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    	}, P<cv::Scalar>(V4D::Keys::CLEAR_COLOR));
    	return self<Plan>();
    }

    cv::Ptr<Plan> capture() {
    	capture([](const cv::UMat& inputFrame, cv::UMat& f){
    		if(!inputFrame.empty())
    			inputFrame.copyTo(f);
    	}, Edge<cv::UMat, false, false>::make(*this, captureFrame_));

        fb([](cv::UMat& framebuffer, const cv::UMat& f) {
        	if(!f.empty()) {
        		if(f.size() != framebuffer.size())
        			resizePreserveAspectRatio(f, framebuffer, framebuffer.size());
        		else
        			f.copyTo(framebuffer);
        	}
        }, Edge<cv::UMat, false, true>::make(*this, captureFrame_));
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> capture(Tfn fn, Args ... args) {
		using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;
		using isconst_t = std::is_const<Tfb>;
		using fbEdge_t = std::disjunction<
			values_equal<isconst_t::value, true, decltype(R_I(runtime_->sourceCtx()->sourceBuffer()))>,
			values_equal<isconst_t::value, false, decltype(RW_I(runtime_->sourceCtx()->sourceBuffer()))>
		>;

		auto srcEdge = makeInternalEdge<std::is_const<Tfb>::value>(runtime_->sourceCtx()->sourceBuffer());
    	auto wrap = wrap_callable<void>(fn, srcEdge, args...);

        const string id = make_id(this->space(), "capture", fn, args...);

		static_assert((std::is_same<Tfb,const cv::UMat>::value) || !"The first argument must be of type 'const cv::UMat&'");
		emit_access(id, R_I(*this));
		(emit_access(id, args ),...);

		std::function<void((
				typename decltype(srcEdge)::ref_t,
				typename Args::ref_t...))> functor(wrap);
		add_transaction(runtime_->sourceCtx(),id, functor, srcEdge, args...);
		return self<Plan>();
    }

    cv::Ptr<Plan> write() {
        fb([](const cv::UMat& framebuffer, cv::UMat& f) {
            framebuffer.copyTo(f);
        }, Edge<cv::UMat, false, false>::make(*this, writerFrame_));

    	write([](cv::UMat& outputFrame, const cv::UMat& f){
   			f.copyTo(outputFrame);
    	}, Edge<cv::UMat, false, true>::make(*this, writerFrame_));
		return self<Plan>();
    }


    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> write(Tfn fn, Args ... args) {
		using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;
		static_assert((std::is_same<Tfb,cv::UMat>::value) || !"The first argument must be of type 'cv::UMat&'");
		auto sinkEdge = makeInternalEdge<std::is_const<Tfb>::value>(runtime_->sinkCtx()->sinkBuffer());
    	auto wrap = wrap_callable<void>(fn, sinkEdge, args...);

        const string id = make_id(this->space(), "write", fn, args...);
		emit_access(id, R_I(*this));
		(emit_access(id, args ),...);


		std::function<void((
				typename decltype(sinkEdge)::ref_t,
				typename Args::ref_t...))> functor(wrap);
		add_transaction(runtime_->sinkCtx(),id, functor, sinkEdge, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> nvg(Tfn fn, Args... args) {
        auto wrap = wrap_callable<void>(fn, args...);

        const string id = make_id(this->space(), "nvg", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->nvgCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> bgfx(Tfn fn, Args... args) {
        auto wrap = wrap_callable<void>(fn, args...);

        const string id = make_id(this->space(), "bgfx", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->bgfxCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> plain(Tfn fn, Args... args) {
        auto wrap = wrap_callable<void>(fn, args...);

        const string id = make_id(this->space(), "plain", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subInfer(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->infer();

    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subSetup(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->setup();

    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subTeardown(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->teardown();

    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

    template<typename Tedge, typename Tkey = decltype(runtime_)::element_type::Keys::Enum>
	typename std::enable_if<std::is_base_of_v<EdgeBase, Tedge>, cv::Ptr<Plan>>::type
	set(Tkey key, Tedge val) {
		auto plan = self<Plan>();
        auto fn = [plan, key](decltype(val.ref()) v){
        	plan->runtime_->set(key, v);
        };

        const string id = make_id(this->space(), "set", fn, val);
        emit_access(id, R_I(*this));
        emit_access(id, val);
        std::function<void(decltype(val.ref()))> functor(fn);
		add_transaction(runtime_->plainCtx(), id, functor, val);
		return self<Plan>();
	}

	template<typename Tfn, typename ... Args, typename Tkey = decltype(runtime_)::element_type::Keys::Enum>
	typename std::enable_if<!std::is_base_of_v<EdgeBase, Tfn>, cv::Ptr<Plan>>::type
	set(Tkey key, Tfn fn, Args ... args) {
		auto wrapInner = wrap_callable(fn, args...);

		const string id = make_id(this->space(), "set-fn", fn, args...);
        emit_access(id, R_I(*this));
        (emit_access(id, args ),...);
        auto plan = self<Plan>();
		std::function wrap = [plan, key, wrapInner](typename Args::ref_t ... values) {
			plan->runtime_->set(key, wrapInner(values...));
		};

        add_transaction(runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
	}

	template<typename TsubPlan, typename Tparent, typename ... Args>
	auto _sub(Tparent* parent, Args&& ... args) {
		return Plan::makeSubPlan<TsubPlan>(parent, std::forward<Args>(args)...);
	}

	template<typename TsubPlan, typename TparentPtr, typename ... Args>
	auto _sub(TparentPtr parent, Args&& ... args) {
		return Plan::makeSubPlan<TsubPlan>(parent.get(), std::forward<Args>(args)...);
	}

	template<typename Tvar>
	void _shared(Tvar& val) {
		Global::instance().registerShared(val);
	}

	template<typename Tfn, typename ... Args>
    void imgui(Tfn fn, Args&& ... args) {
        if(!runtime_->hasImguiCtx())
        	return;

        runtime_->imguiCtx()->build([fn, &args...]() {
        	fn(args...);
		});
    }

	template<typename T>
	detail::Edge<T, false, true> R(T& t) {
		return detail::Edge<T, false, true>::make(*this, t);
	}

	template<typename T>
	detail::Edge<T, false, true, true> RS(T& t) {
		if(!Global::instance().isShared(t)) {
			throw std::runtime_error("You declare a non-shared variable as shared. Maybe you forgot to declare it?.");
		}
		return detail::Edge<T, false, true, true>::make(*this, t, false);
	}

	template<typename T>
	detail::Edge<T, false, false> RW(T& t) {
		return detail::Edge<T, false, false>::make(*this, t);
	}

	template<typename T>
	detail::Edge<T, false, false, true> RWS(T& t) {
		if(!Global::instance().isShared(t)) {
			throw std::runtime_error("You declare a non-shared variable as shared. Maybe you forgot to declare it?.");
		}
		return detail::Edge<T, false, false, true>::make(*this, t, false);
	}

	template<typename T>
	detail::Edge<T, true, true, true> CS(T& t) {
		if(Global::instance().isShared(t)) {
			return detail::Edge<T, true, true, true>::make(*this, t, false);
		} else {
			throw std::runtime_error("You are trying to safe-copy a non-shared variable. Maybe you forgot to declare it?.");
		}
	}

	template<typename T>
	detail::Edge<T, false, false> A(T& t) {
		if constexpr(std::is_const<T>::value) {
			return detail::Edge<T, false, true>::make(*this, t, false);
		} else {
			return detail::Edge<T, false, false>::make(*this, t, false);
		}
	}

	template<typename T>
	detail::Edge<cv::Ptr<T>, false, true, false, T> V(T t) {
		if(Global::instance().isShared(t)) {
			throw std::runtime_error("You declared a shared variable as temporary.");
		}
		auto ptr = cv::makePtr<T>(t);
		return detail::Edge<decltype(ptr), false, true, false, T>::make(*this, ptr, false);
	}

	template<typename Tval, typename Tkey = decltype(runtime_)::element_type::Keys::Enum>
	Property<Tval> P(Tkey key) {
		const auto& ref = runtime_->get<Tval>(key);
		return Property<Tval>(*this, ref);
	}

    template<typename Tplan, typename ... Args>
	static cv::Ptr<Tplan> make(Args&& ... args) {
    	cv::Ptr<Tplan> plan = new Tplan(std::forward<Args>(args)...);
    	plan->self_ = plan;
    	plan->template setActualTypeSize<Tplan>();
		plan->runtime_->set(V4D::Keys::NAMESPACE, plan->space());
		return plan;
    }

    template<typename Tplan, typename ... Args>
	static void run(int32_t workers, Args&& ... args) {
		//for now, if automatic determination of the number of workers is requested,
		//set workers always to 2
		CV_Assert(workers > -2);
		if(workers == -1) {
			workers = 2;
		} else {
			++workers;
		}

		cv::Ptr<Tplan> plan;
		Global& global = Global::instance();
		RunState& state = RunState::instance();

		std::vector<std::thread*> threads;
		{
			static std::mutex runMtx;
			std::lock_guard<std::mutex> lock(runMtx);
			cv::setNumThreads(0);

			if(global.isFirstRun()) {
				global.setMainID(std::this_thread::get_id());
				CV_LOG_INFO(&v4d_tag, "Starting with " << workers << " workers");
			}

	    	plan = make<Tplan>(std::forward<Args>(args)...);

			if(global.isMain()) {
				const string title = plan->runtime_->title();
				auto src = plan->runtime_->getSource();
				auto sink = plan->runtime_->getSink();
				state.set<size_t>(RunState::Keys::WORKERS_STARTED, workers);
				static std::mutex worker_init_mtx_;
				if(!(plan->runtime_->debugFlags() & DebugFlags::DONT_PAUSE_LOG)) {
					CV_LOG_WARNING(&v4d_tag, "Temporary setting log level to warning.");
					cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
				}

				for (size_t i = 0; i < workers; ++i) {
					threads.push_back(
						new std::thread(
							[plan, i, workers, src, sink, &args...] {
								CV_LOG_DEBUG(&v4d_tag, "Creating worker: " << i);
								{
									std::lock_guard guard(worker_init_mtx_);
									cv::Ptr<V4D> worker = V4D::init(*plan->runtime_.get(), plan->runtime_->title() + "-worker-" + std::to_string(i));

									if (src) {
										worker->setSource(src);
									}
									if (sink) {
										worker->setSink(sink);
									}
								}
								Plan::run<Tplan>(0, std::forward<Args>(args)...);
							}
						)
					);
				}
			} else {
				if(V4D::instance()->debugFlags() & DebugFlags::LOWER_WORKER_PRIORITY) {
					CV_LOG_INFO(&v4d_tag, "Lowering worker thread niceness from: " << getpriority(PRIO_PROCESS, gettid()) << " to: " << 1);

					if (setpriority(PRIO_PROCESS, gettid(), 1)) {
						CV_LOG_INFO(&v4d_tag, "Failed to set niceness: " << std::strerror(errno));
					}
				}
			}
		}

		CV_Assert(plan);
		CLExecScope_t scope(plan->runtime_->fbCtx()->getCLExecContext());

		if(global.isMain()) {
			plan->runtime_->printSystemInfo();
		} else {
			try {
				CV_LOG_DEBUG(&v4d_tag, "Setup on worker: " << plan->runtime_->workerIndex());
				plan->setup();
				plan->makeGraph();
				plan->runGraph();
				plan->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Setup failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(&v4d_tag, "Setup finished: " << plan->runtime_->workerIndex());
		}
		if(global.isMain()) {
			try {
				CV_LOG_DEBUG(&v4d_tag, "Loading GUI");
				plan->runtime_->set(V4D::Keys::NAMESPACE, plan->space());
				plan->gui();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Loading GUI failed: %s", ex.what()));
			}
		} else {

			try {
				CV_LOG_DEBUG(&v4d_tag, "Main inference on worker: " << plan->runtime_->workerIndex());
				plan->infer();
				plan->makeGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Main inference failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(&v4d_tag, "Main inference finished: " << plan->runtime_->workerIndex());
		}

		try {
			V4D::run(plan->runtime_, [plan](){
				TimeTracker::getInstance()->execute("iteration", [plan](){
					plan->runGraph();
				});
			});
			CV_LOG_WARNING(&v4d_tag, "Setting loglevel to INFO");
			cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
			CV_LOG_INFO(&v4d_tag, "Starting pipelines with " << state.get<size_t>(RunState::Keys::WORKERS_STARTED) << " workers.");
		} catch(std::exception& ex) {
			CV_Error_(cv::Error::StsError, ("Main plan->runtime_: %s", ex.what()));
		}
		CV_LOG_DEBUG(&v4d_tag, "Main plan->runtime_ finished: " << plan->runtime_->workerIndex());


		if(!global.isMain()) {
			plan->clearGraph();
			CV_LOG_DEBUG(&v4d_tag, "Starting teardown on worker: " << plan->runtime_->workerIndex());
			try {
				plan->teardown();
				plan->makeGraph();
				plan->runGraph();
				plan->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline teardown failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(&v4d_tag, "Teardown complete on worker: " << plan->runtime_->workerIndex());
		} else {
			for(auto& t : threads)
				t->join();
			CV_LOG_INFO(&v4d_tag, "All threads terminated.");
		}
    }
};

} /* namespace v4d */
} /* namespace cv */

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */

