// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_

#include "flags.hpp"
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

	inline std::vector<std::shared_ptr<Joystick>> fetch(const Joystick::Type& t){
		return fetch<Joystick>(t);
	}

	inline std::vector<std::shared_ptr<Keyboard>> fetch(const Keyboard::Type& t){
		return fetch<Keyboard>(t);
	}

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

template<bool read, typename Tfn, typename ... Args>
static auto make_function_ptr(Tfn&& fn, Args ... args) {
	using fun_t = typename edgefun_t<read, typename std::remove_reference<Tfn>::type, Args...>::type;
	return cv::makePtr<fun_t>(std::forward<Tfn>(fn));
}

//template<bool read, typename Tfn, typename ... Args>
//static auto make_function_edge(Tfn&& fn, Args ...) {
//
//	using fun_t = typename edgefun_t<read, typename std::remove_reference<Tfn>::type, Args...>::type;
//
//	return detail::Edge<cv::Ptr<fun_t>, false, read, false>::make(self<Plan>(), cv::makePtr<fun_t>(std::forward<Tfn>(fn)));
//}
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
			FRAMEBUFFER_SIZE,
			STRETCHING,
			CLEAR_COLOR,
			NAMESPACE,
			FULLSCREEN,
			DISABLE_VIDEO_IO,
			DISABLE_INPUT_EVENTS
    	};
    };
private:
    CV_EXPORTS static thread_local std::mutex instance_mtx_;
    CV_EXPORTS static thread_local cv::Ptr<V4D> instance_;
    CV_EXPORTS ThreadSafeAnyMap<Keys::Enum> properties_;

    int32_t workerIdx_ = -1;

    AllocateFlags::Enum allocateFlags_;
    ConfigFlags::Enum configFlags_;
    DebugFlags::Enum  debugFlags_;

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
    		ss << demangle(typeid(decltype(key)).name()) << " = " << size_t(&val) << " (fire: " << fire << ")";
    		CV_LOG_INFO(&mon_tag, ss.str());
    	}
		properties_.set(key, val, fire);
	}

    template<typename Tval>
	const auto& get(Keys::Enum key) const {
		return properties_.get<Tval>(key);
	}

	template <typename V> V apply(Keys::Enum k, std::function<V(V&)> f) {
		return properties_.apply(k, f);
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
    CV_EXPORTS static cv::Ptr<V4D> init(const cv::Rect& viewport, const string& title, AllocateFlags::Enum allocateFlags = AllocateFlags::DEFAULT, ConfigFlags::Enum configFlags = ConfigFlags::DEFAULT, DebugFlags::Enum debugFlags = DebugFlags::DEFAULT, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> init(const cv::Rect& viewport, const cv::Size& fbsize, const string& title, AllocateFlags::Enum allocateFlags = AllocateFlags::DEFAULT, ConfigFlags::Enum configFlags = ConfigFlags::DEFAULT, DebugFlags::Enum debugFlags = DebugFlags::DEFAULT, int samples = 0);
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
    CV_EXPORTS AllocateFlags::Enum allocateFlags();
    CV_EXPORTS ConfigFlags::Enum configFlags();
    CV_EXPORTS DebugFlags::Enum debugFlags();

    static void run(cv::Ptr<V4D> runtime, std::function<void()> runGraph) {
		//the first sequence number is 1!
		static Resequence reseq(1);
    	static std::binary_semaphore frame_sync_render(0);
		static std::binary_semaphore frame_sync_sema_swap(0);
		Global& global = Global::instance();

		try {
			if(global.isMain()) {
				CV_LOG_INFO(&v4d_tag, "Display thread started.");
				while(keep_running()) {
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
				while(keep_running()) {
					bool result = true;
					TimeTracker::getInstance()->execute("worker", [&result, &global, runtime, runGraph](){
						event::poll();
//						if(!runtime->hasSource() || (runtime->hasSource() && !runtime->getSource()->isOpen())) {
						global.apply<size_t>(Global::Keys::RUN_COUNT, [runtime](size_t& s) {
							runtime->setSequenceNumber(++s);
							return s;
						});
//						}

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
		request_finish();
		reseq.finish();
		if(runtime->configFlags() & ConfigFlags::DISPLAY_MODE) {
			if(global.isMain()) {
				for(size_t i = 0; i < global.get<size_t>(Global::Keys::WORKERS_STARTED); ++i)
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
            const string& title, AllocateFlags::Enum allocFlags, ConfigFlags::Enum confFlags, DebugFlags::Enum debFlags, int samples);

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
    friend class SharedVariables;
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
    std::vector<cv::Ptr<Node>> currentNodes_;
    std::vector<cv::Ptr<Node>> allNodes_;
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

    template <typename ... Args, typename Tfn>
    static auto wrap_callable(Tfn fn) {
    	if constexpr(std::is_void<typename CallableTraits<Tfn>::return_type_t>::value || std::is_same<typename CallableTraits<Tfn>::return_type_t, std::false_type>::value) {
			if constexpr(CallableTraits<Tfn>::member_t::value) {
				return std::function([fn](Args... values) -> decltype(std::mem_fn(fn)(values...))  {
					return std::mem_fn(fn)(values...);
				});
			} else {
				return std::function(fn);
			}
    	} else {
			if constexpr(CallableTraits<Tfn>::member_t::value) {
				return std::function([fn](Args... values) ->  decltype(std::mem_fn(fn)(values...))  {
					return std::mem_fn(fn)(values...);
				});
			} else {
				return std::function(fn);
			}
   	   }
   }

	template<bool Tconst, typename T>
    auto makeInternalEdge(T& val) {
		if constexpr(Tconst) {
			return R(val);
		} else {
			return RW(val);
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
    	if(currentNodes_.empty())
    		return;

    	if(currentNodes_.back()->name_ == name)
    		found = currentNodes_.back();
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
     			currentNodes_.push_back(n);
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
			for (auto& n : currentNodes_) {
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
									currentState.condition_ = global.once(n->name_) && n->tx_->performPredicate();
								} else if((btype == BranchType::PARALLEL_ONCE)) {
									currentState.condition_ = !n->tx_->ran() && n->tx_->performPredicate();
								} else {
									CV_Assert(false);
								}
							} else {
								currentState.condition_ = n->tx_->performPredicate();
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
						auto plan = self<Plan>();
						if(lock)
						{
							std::lock_guard<std::mutex> guard(*lock.get());
							auto ctx = n->tx_->getContextCallback()();
							auto viewport = runtime_->get<cv::Rect>(V4D::Keys::VIEWPORT);
							int res = ctx->execute(viewport, [plan, countLockContention, n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [plan, countLockContention, n,currentState](){
//									cerr << "locked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform();
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
							int res = ctx->execute(viewport, [plan, countLockContention, n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [plan, countLockContention, n,currentState](){
//									cerr << "unlocked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform();
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
		//safe all nodes till the end of the plan because they might hold state!
		std::copy(currentNodes_.begin(), currentNodes_.end(), std::back_inserter(allNodes_));

		accesses_.clear();
		branchStateStack_.clear();
		branchStack_.clear();
		transactions_.clear();
		currentNodes_.clear();
	}

	size_t id_ = Global::instance().apply<size_t>(Global::Keys::PLAN_CNT, [](size_t& v){
		++v;
		return v;
	});

	template<typename Tinstance>
    cv::Ptr<Tinstance> self() {
		if(!self_)
			self_ = this;
		return self_.dynamicCast<Tinstance>();
	}

    template<typename Tplan, typename Tparent, typename ... Args>
	static cv::Ptr<Tplan> makeSubPlan(Tparent* parent, Args&& ... args) {
    	Tplan* plan = new Tplan(std::forward<Args>(args)...);
    	plan->setParentID(parent->space());
    	plan->setParentOffset(reinterpret_cast<size_t>(parent));
    	plan->template setParentActualTypeSize<Tparent>();
    	plan->template setActualTypeSize<Tplan>();
    	plan->runtime_->set(V4D::Keys::NAMESPACE, plan->space());
		return plan->template self<Tplan>();
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

//	template<typename ... Args, typename Tkey = decltype(runtime_)::element_type::Keys::Enum>
//	cv::Ptr<Plan> set(std::tuple<Tkey, Args...> t) {
////		auto wrapInner = wrap_callable<typename Args::ref_t ...>(fn);
////
////		const string id = make_id(this->space(), "set-fn", fn, args...);
////        emit_access(id, R(*this));
////        (emit_access(id, args ),...);
////        auto plan = self<Plan>();
////		std::function wrap = [plan, key, wrapInner](typename Args::ref_t ... values) {
////			plan->runtime_->set(key, wrapInner(values...));
////		};
////
////        add_transaction(runtime_->plainCtx(), id, wrap, args...);
//		return self<Plan>();
//	}
public:
//    template<bool read, typename Tfn, typename ... Args>
//    struct edgefun_t<read, std::false_type, Tfn, Args...> {
//    	edgefun_t(Tfn fn, Args ... args) {}
//    	using return_type_t = typename CallableTraits<Tfn>::return_type_t;
//    	static_assert(!std::is_same<return_type_t, std::false_type>::value, "Invalid callable passed");
//    	using type = std::function<return_type_t(typename Args::ref_t ...)>;
//    };


    template<typename T>
	struct Property : detail::Edge<const T, false, true, true> {
		using parent_t = detail::Edge<const T, false, true, true>;
		Property(cv::Ptr<Plan> plan, const T& val) : parent_t(parent_t::make(plan, val)) {
			Global::instance().makeSharedVar(val);
		}
	};

	template<typename TeventClass, typename Tfn = std::function<std::vector<std::shared_ptr<TeventClass>>()>, typename Tparent = Edge<Tfn, false, true, false>>
	struct Event : Tparent {
		Event(cv::Ptr<Plan> plan) : Tparent(Tparent::make(plan, wrap_callable<>([]() {
			if(!V4D::instance()->get<bool>(V4D::Keys::DISABLE_INPUT_EVENTS))
				return cv::v4d::event::fetch<TeventClass>();
			else
				return typename TeventClass::List();
		}))) {
			static_assert(Tparent::func_t::value, "Internal error: Function not recognized!");
		}

		Event(cv::Ptr<Plan> plan, const typename TeventClass::Type t) : Tparent(Tparent::make(plan, wrap_callable<>([t]() {
			if(!V4D::instance()->get<bool>(V4D::Keys::DISABLE_INPUT_EVENTS))
				return cv::v4d::event::fetch(t);
			else
				return typename TeventClass::List();
		}))) {
			static_assert(Tparent::func_t::value, "Internal error: Function not recognized!");
		}

		template<typename Ttrigger>
		Event(cv::Ptr<Plan> plan, const typename TeventClass::Type t, const Ttrigger tr) : Tparent(Tparent::make(plan, wrap_callable<>([t, tr]() {
			if(!V4D::instance()->get<bool>(V4D::Keys::DISABLE_INPUT_EVENTS))
				return cv::v4d::event::fetch(t, tr);
			else
				return typename TeventClass::List();
		}))) {
			static_assert(Tparent::func_t::value, "Internal error: Function not recognized!");
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
		return detail::demangle(typeid(*this).name());
	}

	virtual void setParentID(const string& parent) {
		parent_  = parent;
	}

	virtual std::string getParentID() {
		return parent_;
	}


    template <typename Tfn, typename ... Args>
    typename std::enable_if<!std::is_base_of<EdgeBase, Tfn>::value, cv::Ptr<Plan>>::type
    gl(Tfn fn, Args ... args) {
//    	auto wrap = wrap_callable<typename Args::ref_t...>(fn);
//        const string id = make_id(this->space(), "gl", fn, args...);
//        emit_access(id, R(*this));
//        (emit_access(id, args ),...);
//		add_transaction(runtime_->glCtx(-1), id, wrap, args...);
    	auto argsTuple = std::make_tuple(args...);
    	call(runtime_->glCtx(-1), "gl", fn, std::forward<decltype(argsTuple)>(argsTuple), std::make_index_sequence<std::tuple_size<decltype(argsTuple)>::value>());
    	return self<Plan>();
    }

    template <int32_t pos = 0, typename Tedge, typename Tfn, typename ... Args>
    typename std::enable_if<std::is_base_of<EdgeBase, Tedge>::value, cv::Ptr<Plan>>::type
	gl(Tedge indexEdge, Tfn fn, Args ... args) {
        auto ctxCallback = [this, indexEdge]() {
			Tedge copy = indexEdge;
			return runtime_->glCtx(copy.ref());};
		auto argsTuple = std::make_tuple(args...);
		if constexpr(pos > 0) {
			auto beforePos = sub_tuple<0,pos>(argsTuple);
			auto afterPos = sub_tuple<pos, sizeof...(args) - pos>(argsTuple);
			auto allTuple = std::tuple_cat(beforePos, indexEdge, afterPos);
			return call(ctxCallback, "gl-i", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		} else if constexpr(pos < 0) {
			return call(ctxCallback, "gl-i", fn, std::forward<decltype(argsTuple)>(argsTuple), std::make_index_sequence<std::tuple_size<decltype(argsTuple)>::value>());
		} else {
			auto allTuple = std::make_tuple(indexEdge, args...);
			return call(ctxCallback, "gl-i", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		}
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    typename std::enable_if<!std::is_base_of<EdgeBase, Tfn>::value, cv::Ptr<Plan>>::type
    ext(Tfn fn, Args ... args) {
//    	auto wrap = wrap_callable<typename Args::ref_t...>(fn);
//        const string id = make_id(this->space(), "gl", fn, args...);
//        emit_access(id, R(*this));
//        (emit_access(id, args ),...);
//		add_transaction(runtime_->glCtx(-1), id, wrap, args...);


    	auto argsTuple = std::make_tuple(args...);
    	call(runtime_->extCtx(-1), "ext", fn, std::forward<decltype(argsTuple)>(argsTuple), std::make_index_sequence<std::tuple_size<decltype(argsTuple)>::value>());
    	return self<Plan>();
    }
//    template <typename Tfn, typename ... Args>
//    typename std::enable_if<!std::is_base_of<EdgeBase, Tfn>::value, cv::Ptr<Plan>>::type
//    ext(Tfn fn, Args ... args) {
//    	auto wrap = wrap_callable<typename Args::ref_t...>(fn);
//        const string id = make_id(this->space(), "ext", fn, args...);
//        emit_access(id, R(*this));
//        (emit_access(id, args ),...);
//		add_transaction(runtime_->extCtx(-1), id, wrap, args...);
//		return self<Plan>();
//    }

    template <int32_t pos = 0, typename Tedge, typename Tfn, typename ... Args>
    typename std::enable_if<std::is_base_of<EdgeBase, Tedge>::value, cv::Ptr<Plan>>::type
	ext(Tedge indexEdge, Tfn fn, Args ... args) {
        auto ctxCallback = [this, indexEdge]() {
			Tedge copy = indexEdge;
			return runtime_->extCtx(copy.ref());};
		auto argsTuple = std::make_tuple(args...);
		if constexpr(pos > 0) {
			auto beforePos = sub_tuple<0,pos>(argsTuple);
			auto afterPos = sub_tuple<pos, sizeof...(args) - pos>(argsTuple);
			auto allTuple = std::tuple_cat(beforePos, indexEdge, afterPos);
			return call(ctxCallback, "ext-i", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		} else if constexpr(pos < 0) {
			return call(ctxCallback, "ext-i", fn, std::forward<decltype(argsTuple)>(argsTuple), std::make_index_sequence<std::tuple_size<decltype(argsTuple)>::value>());
		} else {
			auto allTuple = std::make_tuple(indexEdge, args...);
			return call(ctxCallback, "ext-i", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		}
		return self<Plan>();
    }

    template <typename Tedge>
    typename std::enable_if<std::is_base_of_v<EdgeBase, Tedge>, cv::Ptr<Plan>>::type
    branch(Tedge edge) {
        auto wrap = wrap_callable<typename Tedge::ref_t>([](const bool& b){ return b; });
        const string id = make_id(this->space(), "branch", wrap);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R(*this));
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, edge);
		return self<Plan>();
    }

    template <typename Tfn>
    typename std::enable_if<!std::is_base_of_v<EdgeBase, Tfn>, cv::Ptr<Plan>>::type
    branch(Tfn fn) {
        auto wrap = wrap_callable(fn);
        const string id = make_id(this->space(), "branch", fn);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R(*this));
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(Tfn fn, Args ... args) {
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
        const string id = make_id(this->space(), "branch", fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(int workerIdx, Tfn fn, Args ... args) {
        auto wrapInner = wrap_callable<typename Args::ref_t ...>(fn);
        const string id = make_id(this->space(), "branch-pin" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		std::function<bool((typename Args::ref_t...))> wrap = [this, workerIdx, wrapInner](Args ... args){
			return runtime_->workerIndex() == workerIdx && wrapInner(args...);
		};
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tedge>
    typename std::enable_if<std::is_base_of_v<EdgeBase, Tedge>, cv::Ptr<Plan>>::type
    branch(BranchType::Enum type, Tedge edge) {
        auto wrap = wrap_callable<typename Tedge::ref_t>([](const bool& b){ return b; });
        const string id = make_id(this->space(), "branch", wrap);
        branchStack_.push_front({id, type});
        emit_access(id, R(*this));
		add_transaction(type, runtime_->plainCtx(), id, wrap, edge);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(BranchType::Enum type, Tfn fn, Args ... args) {
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
        const string id = make_id(this->space(), "branch-type" + std::to_string((int)type), fn, args...);
        branchStack_.push_front({id, type});
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(type, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> branch(BranchType::Enum type, int workerIdx, Tfn fn, Args ... args) {
        auto wrapInner = wrap_callable<typename Args::ref_t ...>(fn);
        const string id = make_id(this->space(), "branch-type-pin" + std::to_string((int)type) + "-" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, type});
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		std::function<bool((typename Args::ref_t...))> wrap = [this, workerIdx, wrapInner](Args ... args){
			return runtime_->workerIndex() == workerIdx && wrapInner(args...);
		};

		add_transaction(type, runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

/*    template <typename Tfn>
    cv::Ptr<Plan> elseIfBranch(Tfn fn) {
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
    	std::function functor = fn;
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
        (emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
		add_transaction(BranchType::PARALLEL, runtime_->plainCtx(), id, functor, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(int workerIdx, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
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
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
    	std::function functor = fn;
		add_transaction(type, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(BranchType::Enum type, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
    	(emit_access(id, args ),...);
        std::function<bool(typename Args::ref_t...)> functor = fn;
		add_transaction(type, runtime_->plainCtx(), id, functor, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> elseIfBranch(BranchType::Enum type, int workerIdx, Tfn fn, Args ... args) {
//        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
    	emit_access(id, R(*this));
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
        emit_access(id, R(*this));
        std::function functor = [](){ return true; };
		add_transaction(current.second, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    cv::Ptr<Plan> elseBranch() {
    	auto current = branchStack_.front();
    	string id = "[else]" + current.first;
    	emit_access(id, R(*this));
		std::function functor = [](){ return true; };
		add_transaction(current.second, runtime_->plainCtx(), id, functor);
		return self<Plan>();
    }

    template <typename Tctx, typename Tfn, typename Tuple, size_t ... idx>
    cv::Ptr<Plan> call(Tctx ctx, const string& name, Tfn fn, Tuple&& args, std::index_sequence<idx...>) {
		const string id = make_id(this->space(), name, fn, std::get<idx>(args)...);
		emit_access(id, R(*this));
		(emit_access(id, std::get<idx>(args) ),...);
		auto wrap = wrap_callable<typename std::remove_reference<decltype(std::get<idx>(args))>::type::ref_t...>(fn);
		add_transaction(ctx,id, wrap, std::get<idx>(args)...);
		return self<Plan>();
    }

    template <size_t pos = 0, typename Tfn, typename ... Args>
    cv::Ptr<Plan> fb(Tfn fn, Args ... args) {
		using isMemFn = typename CallableTraits<Tfn>::member_t;
		constexpr size_t idx = pos > 0 && isMemFn::value ? pos - 1 : pos;
		using Tfb = typename std::tuple_element<idx, typename function_traits<Tfn>::argument_types>::type;
		auto argsTuple = std::make_tuple(args...);
		if constexpr(pos > 0) {
			auto beforeFb = sub_tuple<0,pos>(argsTuple);
			auto afterFb = sub_tuple<pos, sizeof...(args) - pos>(argsTuple);
			auto fbEdge = std::make_tuple(makeInternalEdge<std::is_const<Tfb>::value>(runtime_->fbCtx()->fb()));
			auto allTuple = std::tuple_cat(beforeFb, fbEdge, afterFb);
			return call(runtime_->fbCtx(), "fb", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		} else {
			auto allTuple = std::make_tuple(makeInternalEdge<std::is_const<Tfb>::value>(runtime_->fbCtx()->fb()), args...);
			return call(runtime_->fbCtx(), "fb", fn, std::forward<decltype(allTuple)>(allTuple), std::make_index_sequence<std::tuple_size<decltype(allTuple)>::value>());
		}
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
    	}, Edge<cv::UMat, false, false>::make(self<Plan>(), captureFrame_));

        fb([](cv::UMat& framebuffer, const cv::UMat& f) {
        	if(!f.empty()) {
        		if(f.size() != framebuffer.size())
        			resize_preserving_aspect_ratio(f, framebuffer, framebuffer.size());
        		else
        			f.copyTo(framebuffer);
        	}
        }, Edge<cv::UMat, false, true>::make(self<Plan>(), captureFrame_));
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> capture(Tfn fn, Args ... args) {
		auto srcEdge = makeInternalEdge<true>(runtime_->sourceCtx()->sourceBuffer());
    	auto wrap = wrap_callable<typename decltype(srcEdge)::ref_t, typename Args::ref_t...>(fn);

        const string id = make_id(this->space(), "capture", fn, args...);

		emit_access(id, R(*this));
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
        }, Edge<cv::UMat, false, false>::make(self<Plan>(), writerFrame_));

    	write([](cv::UMat& outputFrame, const cv::UMat& f){
   			f.copyTo(outputFrame);
    	}, Edge<cv::UMat, false, true>::make(self<Plan>(), writerFrame_));
		return self<Plan>();
    }


    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> write(Tfn fn, Args ... args) {
		using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;
		static_assert((std::is_same<Tfb,cv::UMat>::value) || !"The first argument must be of type 'cv::UMat&'");
		auto sinkEdge = makeInternalEdge<std::is_const<Tfb>::value>(runtime_->sinkCtx()->sinkBuffer());
    	auto wrap = wrap_callable<typename decltype(sinkEdge)::ref_t, typename Args::ref_t...>(fn);

        const string id = make_id(this->space(), "write", fn, args...);
		emit_access(id, R(*this));
		(emit_access(id, args ),...);


		std::function<void((
				typename decltype(sinkEdge)::ref_t,
				typename Args::ref_t...))> functor(wrap);
		add_transaction(runtime_->sinkCtx(),id, functor, sinkEdge, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> nvg(Tfn fn, Args... args) {
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);

        const string id = make_id(this->space(), "nvg", fn, args...);
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->nvgCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<Plan> bgfx(Tfn fn, Args... args) {
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);

        const string id = make_id(this->space(), "bgfx", fn, args...);
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->bgfxCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename ... Args>
    cv::Ptr<Plan> plain(Args... args) {
        auto fn = [](typename Args::ref_t ...){};
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);
        const string id = make_id(this->space(), "plain", wrap, args...);
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename Tfn, typename ... Args>
    typename std::enable_if<!std::is_base_of<EdgeBase, Tfn>::value, cv::Ptr<Plan>>::type
    plain(Tfn fn, Args... args) {
        auto wrap = wrap_callable<typename Args::ref_t ...>(fn);

        const string id = make_id(this->space(), "plain", fn, args...);
        emit_access(id, R(*this));
        (emit_access(id, args ),...);
		add_transaction(runtime_->plainCtx(), id, wrap, args...);
		return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subInfer(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->infer();
    	subPlan->makeGraph();
    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subSetup(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->setup();
    	subPlan->makeGraph();
    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

    template <typename TsubPlan>
    cv::Ptr<Plan> subTeardown(cv::Ptr<TsubPlan> subPlan) {
    	//FIXME check inheritance pattern
    	subPlan->teardown();
      	subPlan->makeGraph();
    	std::copy(subPlan->accesses_.begin(), subPlan->accesses_.end(), std::inserter(accesses_, accesses_.end()));
    	std::copy(subPlan->transactions_.begin(), subPlan->transactions_.end(), std::inserter(transactions_, transactions_.end()));
    	subPlan->clearGraph();
    	return self<Plan>();
    }

//    template<typename Tedge>
//	typename std::enable_if<std::is_base_of_v<EdgeBase, Tedge>, cv::Ptr<Plan>>::type
//	set(decltype(runtime_)::element_type::Keys::Enum key, Tedge val) {
//		auto plan = self<Plan>();
//        auto fn = [plan, key](decltype(val.ref()) v){
//        	plan->runtime_->set(key, v);
//        };
//
//        const string id = make_id(this->space(), "set", fn, val);
//        emit_access(id, R(*this));
//        emit_access(id, val);
//        std::function<void(decltype(val.ref()))> functor(fn);
//		add_transaction(runtime_->plainCtx(), id, functor, val);
//		return self<Plan>();
//	}
//
//	template<typename Tfn, typename ... Args>
//	typename std::enable_if<!std::is_base_of_v<EdgeBase, Tfn>, cv::Ptr<Plan>>::type
//	set(decltype(runtime_)::element_type::Keys::Enum key, Tfn fn, Args ... args) {
//		auto wrapInner = wrap_callable<typename Args::ref_t ...>(fn);
//
//		const string id = make_id(this->space(), "set-fn", fn, args...);
//        emit_access(id, R(*this));
//        (emit_access(id, args ),...);
//        auto plan = self<Plan>();
//		std::function wrap = [plan, key, wrapInner](typename Args::ref_t ... values) {
//			plan->runtime_->set(key, wrapInner(values...));
//		};
//
//        add_transaction(runtime_->plainCtx(), id, wrap, args...);
//		return self<Plan>();
//	}


	template <typename Tkey, typename Tfn, typename Ttuple, size_t ... idx>
	auto make_setter_function(Tkey key, Tfn fn, Ttuple&& args, std::index_sequence<idx...>) {
		auto plan = self<Plan>();
		return std::function([plan, key, fn](decltype(std::get<idx>(args).ref()) ... values){
        	plan->runtime_->set(key, fn(values...));
        });
	}

	template <typename Ttuple>
	auto make_setter_function(Ttuple&& values) {
		using tuple_t = typename std::remove_reference<Ttuple>::type;
		constexpr size_t sz = std::tuple_size<tuple_t>::value;
		static_assert(std::is_enum<typename std::tuple_element<0, tuple_t>::type>::value, "Can not set a property without a key as first argument");
		static_assert(sz > 1, "Can not set a property without value");
		auto key = std::get<0>(values);
		auto val = std::get<1>(values);
		if constexpr(!is_callable<decltype(val)>::value) {
			static_assert(sz == 2, "Can not set a Property from multiple Edges");
			auto plan = self<Plan>();
			return std::function([plan, key](decltype(val.ref()) v){
	        	plan->runtime_->set(key, v);
	        });
		} else {

	////		static_assert(, "The first argument after the key must either be an Edge or a Callable");
			auto fn = std::get<1>(values);
			auto args = sub_tuple<1, sz - 1>(values);
			auto plan = self<Plan>();
			return make_setter_function(key, fn, args, std::make_index_sequence<sz - 2>());
		}
	}

	template <typename TwrapFn, typename Ttuple, size_t ... idx>
	cv::Ptr<Plan> set(const string& id, TwrapFn wrap, Ttuple&& args, std::index_sequence<idx...>) {
        emit_access(id, R(*this));
        (emit_access(id, std::get<idx>(args)),...);
        add_transaction(runtime_->plainCtx(), id, wrap, std::get<idx>(args)...);
		return self<Plan>();
	}


	template <typename Tedge, size_t ... idx>
	cv::Ptr<Plan> set(std::tuple<V4D::Keys::Enum,Tedge>&& values, std::index_sequence<idx...>) {
		const string id = make_id(this->space(), "set-fn", std::get<idx>(values)...);
		std::function wrap = make_setter_function(values);
		auto args = sub_tuple<1, std::tuple_size<std::tuple<V4D::Keys::Enum,Tedge>>::value - 1>(values);
		return set(id, wrap, std::forward<decltype(args)>(args), std::make_index_sequence<std::tuple_size<decltype(args)>::value>());
	}

	template <typename Tedge>
	cv::Ptr<Plan> set(std::tuple<V4D::Keys::Enum,Tedge>&& values) {
		using sz = std::tuple_size<std::tuple<V4D::Keys::Enum,Tedge>>;
		return set(std::forward<std::tuple<V4D::Keys::Enum, Tedge>>(values), std::make_index_sequence<sz::value>());
	}

	template<typename Tedge>
	cv::Ptr<Plan> set(const V4D::Keys::Enum& key, const Tedge& e) {
		return set(std::make_tuple(key, e), std::make_index_sequence<1>());
	}

	template<typename ... Args>
	cv::Ptr<Plan> set(std::tuple<V4D::Keys::Enum,Args>&& ... tuples) {
		(set(std::forward<std::tuple<V4D::Keys::Enum,Args>>(tuples)),...);
		return self<Plan>();
	}

//    template<typename TdstEdge, typename TfnOp, typename TsrcEdge>
//	typename std::enable_if<std::is_base_of_v<EdgeBase, TdstEdge> && std::is_base_of_v<EdgeBase, TsrcEdge>  && is_callable<TfnOp>::value, cv::Ptr<Plan>>::type
//	op(TdstEdge dst, TfnOp op, TsrcEdge src) {
//    	auto fn = [op](decltype(dst.ref()) d, decltype(src.ref()) s){
//        	op(d, s);
//        };
//
//        const string id = make_id(this->space(), "op", op, src);
//        emit_access(id, R(*this));
//        emit_access(id, dst);
//        emit_access(id, src);
//        std::function<void(decltype(dst.ref()), decltype(src.ref()))> functor(fn);
//		add_transaction(runtime_->plainCtx(), id, functor, dst, src);
//		return self<Plan>();
//	}

//	template<typename TfnOp, typename Tfn, typename ... Args>
//	auto
//	uop(TfnOp op, Tfn srcFn, Args ... args) {
//		auto wrapSrc = wrap_callable(srcFn, args.ref()...);
//		using ret_src_t = typename CallableTraits<decltype(wrapSrc)>::return_type_t;
//		auto wrapOp = wrap_callable<std::false_type, TfnOp, void, ret_src_t>(op);
//		using ret_op_t = typename CallableTraits<decltype(wrapOp)>::return_type_t;
//		static_assert(!std::is_same<ret_src_t, std::false_type>::value, "Invalid src callable passed to Plan::op");
//		static_assert(!std::is_same<ret_op_t, std::false_type>::value, "Invalid op callable passed to Plan::op");
//
//		constexpr bool hasReturn = !std::is_same<ret_op_t, void>::value;
//		using val_t = typename std::disjunction<
//						values_equal<hasReturn, true, ret_op_t>,
//						default_type<int>
//					>::type;
//
//		cv::Ptr<val_t> retPtr;
//		if constexpr(hasReturn) {
//			retPtr = new ret_op_t();
//		}
//
//
//		const string id = make_id(this->space(), "unary-op", srcFn, args...);
//        emit_access(id, R(*this));
//        (emit_access(id, args ),...);
//
//		std::function wrap = [retPtr, hasReturn, wrapOp, wrapSrc](typename Args::ref_t ... values) {
//			if constexpr(hasReturn) {
//				(*retPtr.get()) = wrapOp(wrapSrc(values...));
//			} else {
//				wrapOp(wrapSrc(values...));
//			}
//
//		};
//
//		add_transaction(runtime_->plainCtx(), id, wrap, args...);
//		if constexpr(hasReturn) {
//			return detail::Edge<decltype(retPtr), false, true, false, val_t>::make(self<Plan>(), retPtr);
//		} else {
//			return self<Plan>();
//		}
//	}

	template<bool TmakeEdge = true, typename TfnOp, typename ... Args>
	auto make_op(TfnOp fn, Args ... args) {
		auto op = wrap_callable<typename Args::ref_t ...>(fn);
		using ret_t = typename CallableTraits<decltype(op)>::return_type_t;
		constexpr bool hasReturn = !std::is_same<ret_t, void>::value;
//		static_assert(hasReturn || !TmakeEdge, "Operators may not have a return type of void.");
		using ret_no_ref_t = typename std::remove_reference<ret_t>::type;
		static_assert(!std::is_same<ret_no_ref_t, std::false_type>::value, "Invalid callable passed to Plan::op");
		constexpr bool returnsRef = std::is_lvalue_reference<ret_t>::value;
		constexpr bool returnsPtr = std::is_pointer<ret_no_ref_t>::value;

		using val_t = typename std::disjunction<
						values_equal<hasReturn, true, typename std::remove_pointer<ret_no_ref_t>::type>,
						default_type<int>
					>::type;


		if constexpr(hasReturn && TmakeEdge) {
			cv::Ptr<cv::Ptr<val_t>> retPtr = new cv::Ptr<val_t>(cv::Ptr<val_t>(), nullptr);
			std::function wrap = [op](cv::Ptr<val_t>& v, typename Args::ref_t ... values) mutable {
				if constexpr(returnsPtr) {
					v = cv::Ptr<val_t>(cv::Ptr<val_t>(),op(values...));
				} else if constexpr(returnsRef) {
					auto& ref = op(values...);
					v = cv::Ptr<val_t>(cv::Ptr<val_t>(),std::addressof(ref));
				} else {
					v = cv::Ptr<val_t>(cv::Ptr<val_t>(),new val_t(op(values...)));
				}
			};

			const string id = make_id(this->space(), "nary-op", wrap, args...);
			emit_access(id, R(*this));
			(emit_access(id, args ),...);

			auto ptrEdge = detail::Edge<cv::Ptr<cv::Ptr<val_t>>, false, false, false, cv::Ptr<val_t>, true>::make(self<Plan>(), retPtr);
			add_transaction(runtime_->plainCtx(), id, wrap, ptrEdge, args...);

			return detail::Edge<cv::Ptr<val_t>, false, false, false, val_t>::make(self<Plan>(), *retPtr.get());
		} else {
			std::function wrap = [op](typename Args::ref_t ... values) {
				op(values...);
			};

			const string id = make_id(this->space(), "nary-op", wrap, args...);
			emit_access(id, R(*this));
			(emit_access(id, args ),...);
			add_transaction(runtime_->plainCtx(), id, wrap, args...);
			return self<Plan>();
		}
	}



//	template<typename TdstEdge, typename TfnOp, typename ... Args>
//	auto op(TdstEdge dst, TfnOp op, Args ... args) {
//		using ret_t = typename CallableTraits<TfnOp>::return_type_t;
//		static_assert(!std::is_same<ret_t, std::false_type>::value, "Invalid callable passed to Plan::op");
//		constexpr bool hasReturn = !std::is_same<ret_t, void>::value;
//		using val_t = typename std::disjunction<
//						values_equal<hasReturn, true, ret_t>,
//						default_type<int>
//					>::type;
//
//		cv::Ptr<val_t> retPtr;
//		if constexpr(hasReturn) {
//			retPtr = new ret_t();
//		}
//
//		std::function wrap = [retPtr,op](typename TdstEdge::ref_t d, typename Args::ref_t ... values) {
//			if constexpr(hasReturn) {
//				(*retPtr.get()) = op(d, values...);
//			} else {
//				op(d, values...);
//			}
//		};
//
//		const string id = make_id(this->space(), "op", wrap, dst, args...);
//		emit_access(id, R(*this));
//		(emit_access(id, args ),...);
//		(emit_access(id, dst ));
//
//
//		add_transaction(runtime_->plainCtx(), id, wrap, dst, args...);
//		if constexpr(hasReturn) {
//			return detail::Edge<decltype(retPtr), false, true, false, val_t>::make(self<Plan>(), retPtr);
//		} else {
//			return self<Plan>();
//		}
//	}


	template<Operators Top, typename ... Edges>
	cv::Ptr<Plan> op(Edges ... edges){
		return make_op<false>(make_operator_func<check_op<Top, Edges...>::value>(edges...), edges...);
	}

	template<typename ... Edges>
	cv::Ptr<Plan> assign(Edges ... edges){
		return make_op<false>(make_operator_func<check_op<ASSIGN_, Edges...>::value>(edges...), edges...);
	}

	template<typename ... Edges>
	cv::Ptr<Plan> construct(Edges ... edges){
		return make_op<false>(make_operator_func<check_op<CONSTRUCT_, Edges...>::value>(edges...), edges...);
	}

	template<Operators Top, typename ... Edges>
	auto OP(Edges ... edges){
		return make_op(make_operator_func<check_op<Top, Edges...>::value>(edges...), edges...);
	}

	template<typename ... Edges>
	auto operator()(Edges&& ... edges){
		return OP<Operators::CONSTRUCT_>(edges...);
	}

	template<typename ... Edges>
	auto IF(Edges&& ... edges){
		return OP<Operators::IF_>(edges...);
	}

	template<typename ... Edges>
	auto ASSIGN(Edges&& ... edges){
		return OP<Operators::ASSIGN_>(edges...);
	}

	template<typename ... Edges>
	auto SUB(Edges&& ... edges){
		return OP<Operators::SUB_>(edges...);
	}

	template<typename ... Edges>
	auto MUL(Edges&& ... edges){
		return OP<Operators::MUL_>(edges...);
	}

	template<typename ... Edges>
	auto DIV(Edges&& ... edges){
		return OP<Operators::DIV_>(edges...);
	}

	template<typename ... Edges>
	auto MOD(Edges&& ... edges){
		return OP<Operators::MOD_>(edges...);
	}

	template<typename ... Edges>
	auto INCL(Edges&& ... edges){
		return OP<Operators::INCL_>(edges...);
	}

	template<typename ... Edges>
	auto INCR(Edges&& ... edges){
		return OP<Operators::INCR_>(edges...);
	}

	template<typename ... Edges>
	auto DECL(Edges&& ... edges){
		return OP<Operators::DECL_>(edges...);
	}

	template<typename ... Edges>
	auto DECR(Edges&& ... edges){
		return OP<Operators::DECR_>(edges...);
	}

	template<typename ... Edges>
	auto AND(Edges&& ... edges){
		return OP<Operators::AND_>(edges...);
	}

	template<typename ... Edges>
	auto OR(Edges&& ... edges){
		return OP<Operators::OR_>(edges...);
	}

	template<typename ... Edges>
	auto EQ(Edges&& ... edges){
		return OP<Operators::EQ_>(edges...);
	}

	template<typename ... Edges>
	auto NEQ(Edges&& ... edges){
		return OP<Operators::NEQ_>(edges...);
	}

	template<typename ... Edges>
	auto LT(Edges&& ... edges){
		return OP<Operators::LT_>(edges...);
	}

	template<typename ... Edges>
	auto GT(Edges&& ... edges){
		return OP<Operators::GT_>(edges...);
	}

	template<typename ... Edges>
	auto LE(Edges&& ... edges){
		return OP<Operators::LE_>(edges...);
	}

	template<typename ... Edges>
	auto GE(Edges&& ... edges){
		return OP<Operators::GE_>(edges...);
	}

	template<typename ... Edges>
	auto NOT(Edges&& ... edges){
		return OP<Operators::NOT_>(edges...);
	}

	template<typename ... Edges>
	auto XOR(Edges&& ... edges){
		return OP<Operators::XOR_>(edges...);
	}

	template<typename ... Edges>
	auto BAND(Edges&& ... edges){
		return OP<Operators::BAND_>(edges...);
	}

	template<typename ... Edges>
	auto BOR(Edges&& ... edges){
		return OP<Operators::BOR_>(edges...);
	}


	template<typename ... Edges>
	auto SHL(Edges&& ... edges){
		return OP<Operators::SHL_>(edges...);
	}

	template<typename ... Edges>
	auto SHR(Edges&& ... edges){
		return OP<Operators::SHR_>(edges...);
	}

	template<typename Tfn, typename ... Args>
	auto F(Tfn src, Args&& ... args) {
		return make_op(src, args...);
	}


//	template<typename Tret, auto fn, typename ... Args, typename Tsrc = Tret(cv::UMat::*)(typename Args::element_type_t...)>
//	constexpr auto F(Args ... args) {
//		return static_cast<Tsrc>(fn);
//	}

//	constexpr auto copyToMemFn = static_cast<void(*)(const cv::UMat&, cv::UMat&)>(&SharedVariables::copy);
	template<typename ... Args>
	auto _(Args&& ... args) {
		return std::make_tuple(std::forward<const Args>(args)...);
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
	void _safe(Tvar& val) {
		Global::instance().registerSafe(val);
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
	detail::Edge<T, false, true> R(const T& t) {
		return detail::Edge<T, false, true>::make(self<Plan>(), t);
	}

	template<typename T>
	detail::Edge<T, false, true, true> RS(const T& t) {
		if(!Global::instance().checkShared(*this, t)) {
			throw std::runtime_error("You declare a non-shared variable as shared. Maybe you forgot to declare it?.");
		}
		return detail::Edge<T, false, true, true>::make(self<Plan>(), t);
	}

	template<typename T>
	detail::Edge<T, false, false> RW(T& t) {
		return detail::Edge<T, false, false>::make(self<Plan>(), t);
	}

	template<typename T>
	detail::Edge<T, false, false, true> RWS(T& t) {
		if(!Global::instance().checkShared(*this, t)) {
			throw std::runtime_error("You declare a non-shared variable as shared. Maybe you forgot to declare it?.");
		}
		return detail::Edge<T, false, false, true>::make(self<Plan>(), t);
	}

	template<typename T>
	detail::Edge<T, true, true, true> CS(T& t) {
		if(Global::instance().checkShared(*this, t)) {
			return detail::Edge<T, true, true, true>::make(self<Plan>(), t);
		} else {
			throw std::runtime_error("You are trying to safe-copy a non-shared variable. Maybe you forgot to declare it?.");
		}
	}

	template<typename T>
	detail::Edge<cv::Ptr<T>, false, true, false, T, true> V(T t) {
		auto ptr = cv::makePtr<T>(t);
		return detail::Edge<decltype(ptr), false, true, false, T, true>::make(self<Plan>(), ptr);
	}

	template<typename Tval>
	Property<Tval> P(V4D::Keys::Enum key) {
		const auto& ref = runtime_->get<Tval>(key);
		return Property<Tval>(self<Plan>(), ref);
	}

	template<typename Tval>
	Property<Tval> P(RunState::Keys::Enum key) {
		const auto& ref = RunState::instance().get<Tval>(key);
		return Property<Tval>(self<Plan>(), ref);
	}

	template<typename Tval>
	Property<Tval> P(Global::Keys::Enum key) {
		const auto& ref = Global::instance().get<Tval>(key);
		return Property<Tval>(self<Plan>(), ref);
	}

	template<typename Tclass>
	Event<Tclass> E() {
		return Event<Tclass>(self<Plan>());
	}

	template<typename Tclass>
	Event<Tclass> E(typename Tclass::Type t) {
		return Event<Tclass>(self<Plan>(), t);
	}

	template<typename Tclass, typename Ttrigger>
	Event<Tclass> E(typename Tclass::Type t, Ttrigger tr) {
		return Event<Tclass>(self<Plan>(), t, tr);
	}

	template<typename Tplan, typename ... Args>
	static cv::Ptr<Tplan> make(Args&& ... args) {
    	Tplan* plan = new Tplan(std::forward<Args>(args)...);
    	plan->template setActualTypeSize<Tplan>();
		plan->runtime_->set(V4D::Keys::NAMESPACE, plan->space());
		return plan->template self<Tplan>();
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
				global.set<size_t>(Global::Keys::WORKERS_STARTED, workers);
//				std::cerr << "workers: " << global.get<size_t>(Global::Keys::WORKERS_STARTED) << std::endl;
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
#if defined(__linux__)
					CV_LOG_INFO(&v4d_tag, "Lowering worker thread niceness from: " << getpriority(PRIO_PROCESS, gettid()) << " to: " << 1);

					if (setpriority(PRIO_PROCESS, gettid(), 1)) {
						CV_LOG_INFO(&v4d_tag, "Failed to set niceness: " << std::strerror(errno));
					}
#endif
				}
			}
		}

		CV_Assert(plan);

		if(global.isMain()) {
			plan->runtime_->printSystemInfo();
		} else {
			static std::binary_semaphore setup_sema(1);
			try {
				CV_LOG_DEBUG(&v4d_tag, "Setup on worker: " << plan->runtime_->workerIndex());
				setup_sema.acquire();
				plan->setup();
				plan->makeGraph();
				plan->runGraph();
				plan->clearGraph();
				setup_sema.release();
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
			CV_LOG_INFO(&v4d_tag, "Starting pipelines with " << global.get<size_t>(Global::Keys::WORKERS_STARTED) << " workers.");
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
			V4D::instance()->setSink(nullptr);
			V4D::instance()->setSource(nullptr);
			CV_LOG_DEBUG(&v4d_tag, "Teardown complete on worker: " << plan->runtime_->workerIndex());
		} else {
			for(auto& t : threads)
				t->join();
			V4D::instance()->setSink(nullptr);
			V4D::instance()->setSource(nullptr);
			CV_LOG_INFO(&v4d_tag, "All threads terminated.");
		}
    }
};

//template<typename Tfirst, typename ... Edges>
//typename std::enable_if<(std::is_base_of<EdgeBase, Edges>::value && ...), std::tuple<Tfirst, Edges...>>::type
//operator,(Tfirst&& first, Edges&& ... edges){
//	return std::make_tuple<Tfirst, Edges...>(std::forward<Tfirst>(first), std::forward<Edges...>(edges...));
//}




template<typename ... Edges>
auto operator+(const std::tuple<Edges...>& tuple){
	return Operation::op<ADD_>(tuple);
}

template<typename TedgeL, typename ... Edges>
auto operator+(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<ADD_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator+(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator+(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator+(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator+(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator+(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator+(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}


template<typename TedgeL, typename ... Edges>
auto operator*(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<MUL_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator*(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator*(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator*(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator*(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator*(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator*(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator-(const std::tuple<Edges...>& tuple){
	return Operation::op<SUB_>(tuple);
}

template<typename TedgeL, typename ... Edges>
auto operator-(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<SUB_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator-(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator-(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator-(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator-(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator-(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator-(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}
template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator-(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator*(rhs, std::make_tuple(std::forward<decltype(rhs.plan()->V(-1))>(rhs.plan()->V(-1))));
}

template<typename T>
auto operator-(const Plan::Property<T>& rhs){
	return operator*(rhs, std::make_tuple(std::forward<decltype(rhs.plan()->V(-1))>(rhs.plan()->V(-1))));
}

template<typename T>
auto operator-(const Plan::Event<T>& rhs){
	return operator*(rhs, std::make_tuple(std::forward<decltype(rhs.plan()->V(-1))>(rhs.plan()->V(-1))));
}


template<typename TedgeL, typename ... Edges>
auto operator/(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<DIV_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator/(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator/(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator/(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator/(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator/(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator/(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}
template<typename TedgeL, typename ... Edges>
auto operator%(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<MOD_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator%(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator%(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator%(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator%(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator%(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator%(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator++(const std::tuple<Edges...>& tuple){
	return Operation::op<INCL_>(tuple);
}

template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator++(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& e){
	return operator++(std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(e)));
}

template<typename T>
auto operator++(const Plan::Property<T>& rhs){
	return operator--(std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename T>
auto operator++(const Plan::Event<T>& rhs){
	return operator--(std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator++(const std::tuple<Edges...>& tuple, int){
	return Operation::op<INCR_>(tuple);
}

template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator++(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& e, int){
	return operator++(std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(e)));
}

template<typename T>
auto operator++(const Plan::Property<T>& rhs, int){
	return operator++(std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename T>
auto operator++(const Plan::Event<T>& rhs, int){
	return operator++(std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator--(const std::tuple<Edges...>& tuple){
	return Operation::op<DECL_>(tuple);
}

template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator--(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& e){
	return operator--(std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(e)));
}

template<typename T>
auto operator--(const Plan::Property<T>& rhs){
	return operator--(std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename T>
auto operator--(const Plan::Event<T>& rhs){
	return operator--(std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator--(const std::tuple<Edges...>& tuple, int){
	return Operation::op<DECR_>(tuple);
}

template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator--(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& e, int){
	return operator--(std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(e)));
}

template<typename T>
auto operator--(const Plan::Property<T>& rhs, int){
	return operator--(std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename T>
auto operator--(const Plan::Event<T>& rhs, int){
	return operator--(std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator&&(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<AND_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator&&(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator&&(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator&&(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator&&(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator&&(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator&&(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}


template<typename TedgeL, typename ... Edges>
auto operator||(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<OR_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator||(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator||(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator||(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator||(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator||(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator||(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator==(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<EQ_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator==(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator==(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator==(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator==(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator==(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator==(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator!=(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<NEQ_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator!=(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator!=(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator!=(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator!=(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator!=(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator!=(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}
template<typename TedgeL, typename ... Edges>
auto operator<(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<LT_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator<(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator<(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator<(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator<(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator>(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<GT_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator>(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator>(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator>(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator>(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}
template<typename TedgeL, typename ... Edges>
auto operator<=(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<LE_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator<=(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator<=(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<=(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator<=(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<=(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator<=(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator>=(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<GE_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator>=(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator>=(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>=(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator>=(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>=(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator>=(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename ... Edges>
auto operator!(const std::tuple<Edges...>& tuple){
	return Operation::op<NOT_>(tuple);
}

template<typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator!(const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& e){
	return operator!(std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(e)));
}

template<typename T>
auto operator!(const Plan::Property<T>& rhs){
	return operator!(std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename T>
auto operator!(const Plan::Event<T>& rhs){
	return operator!(std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator^(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<XOR_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator^(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator^(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator^(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator^(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator^(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator^(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator&(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<BAND_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator&(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator&(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator&(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator&(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator&(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator&(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator|(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<BAND_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator|(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator|(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator|(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator|(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator|(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator|(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator<<(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<SHL_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator<<(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator<<(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<<(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator<<(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator<<(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator<<(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

template<typename TedgeL, typename ... Edges>
auto operator>>(const TedgeL& lhs, const std::tuple<Edges...>& tuple){
	return Operation::op<SHR_>(std::tuple_cat(std::make_tuple(std::forward<const TedgeL>(lhs)), tuple));
}

template<typename TedgeL, typename Telement, bool Tcopy, bool Tread, bool Tshared, typename Tbase, bool TbyValue>
auto operator>>(const TedgeL& lhs, const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>& rhs){
	return operator>>(lhs, std::make_tuple(std::forward<const Edge<Telement, Tcopy, Tread, Tshared, Tbase, TbyValue>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>>(const TedgeL& lhs, const Plan::Property<T>& rhs){
	return operator>>(lhs, std::make_tuple(std::forward<const Plan::Property<T>>(rhs)));
}

template<typename TedgeL, typename T>
auto operator>>(const TedgeL& lhs, const Plan::Event<T>& rhs){
	return operator>>(lhs, std::make_tuple(std::forward<const Plan::Event<T>>(rhs)));
}

} /* namespace v4d */
} /* namespace cv */

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */

