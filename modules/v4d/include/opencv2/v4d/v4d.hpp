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
#include "threadsafemap.hpp"
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
#define EVENT_API_EXPORT CV_EXPORTS
#include "events.hpp"

#include <sys/resource.h>
#include <sys/types.h>
#include <type_traits>
#include <shared_mutex>
#include <iostream>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <barrier>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using namespace std::chrono_literals;

/*!
 * OpenCV namespace
 */
namespace cv {
/*!
 * V4D namespace
 */
namespace v4d {

namespace detail {
template<typename T, bool Tlocked, bool Tcopy, bool Tread>
class Edge {
	using locked_t = std::integral_constant<bool, Tlocked>;
	using copy_t = std::integral_constant<bool, Tcopy>;
	using read_t = std::integral_constant<bool, Tread>;

	T* ptr_ = nullptr;
	const T* const_ptr_ = nullptr;

	static void check() {
		static_assert(locked_t::value && copy_t::value, "You can't lock and copy a value at the same time");
		static_assert(!read_t::value && copy_t::value, "You can't copy a write-able value");
	}

	void set(typename std::enable_if<Tread, const T&>::type t) {
		const_ptr_ = &t;
	}

	void set(typename std::enable_if<!Tread, T&>::type t) {
		ptr_ = &t;
	}
public:
	static Edge make(T& t) {
		check();
		Edge e;
		e.set(t);
		return e;
	}

	static Edge make(const T& t) {
		check();
		Edge e;
		e.set(t);
		return e;
	}


	typename std::enable_if<Tread, const T&>::type get() {
		return *const_ptr_;
	}

	typename std::enable_if<!Tread, T&>::type get() {
		return *ptr_;
	}
};

}

template<typename T>
auto& R(const T& t) {
	return detail::Edge<T, false, false, true>::make(t);
}

template<typename T>
auto& R_COPY(const T& t) {
	return detail::Edge<T, false, true, true>::make(t);
}

template<typename T>
auto& R_LOCK(const T& t) {
	if(!Global::isShared(t)) {
		throw std::runtime_error("You are trying to lock a non-shared variable.");
	}
	return detail::Edge<T, true, false, true>::make(t);
}

template<typename T>
auto& W(T& t) {
	return detail::Edge<T, false, false, false>::make(t);
}

template<typename T>
auto& W_LOCK(T& t) {
	if(!Global::isShared(t)) {
		throw std::runtime_error("You are trying to lock a non-shared variable.");
	}
	return detail::Edge<T, true, false, false>::make(t);
}

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
		DEFAULT = NANOVG | IMGUI | BGFX
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
	};
};

class Plan {
	const cv::Size sz_;
	const cv::Rect vp_;
	std::string parent_;
public:

	//predefined branch predicates
	constexpr static auto always_ = []() { return true; };
	constexpr static auto isTrue_ = [](const bool& b) { return b; };
	constexpr static auto isFalse_ = [](const bool& b) { return !b; };
	constexpr static auto and_ = [](const bool& a, const bool& b) { return a && b; };
	constexpr static auto or_ = [](const bool& a, const bool& b) { return a || b; };

	explicit Plan(const cv::Rect& vp) : sz_(cv::Size(vp.width, vp.height)), vp_(vp){};

	virtual ~Plan() {};

	virtual void gui(cv::Ptr<V4D> window) { CV_UNUSED(window); };
	virtual void setup(cv::Ptr<V4D> window) { CV_UNUSED(window); };
	virtual void infer(cv::Ptr<V4D> window) = 0;
	virtual void teardown(cv::Ptr<V4D> window) { CV_UNUSED(window); };

	virtual std::string id() {
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

	const cv::Size& size() const {
		return sz_;
	}
	const cv::Rect& viewport() const {
		return vp_;
	}
};
/*!
 * Private namespace
 */
namespace detail {

template <typename T> using static_not = std::integral_constant<bool, !T::value>;

//https://stackoverflow.com/a/34873353/1884837
template<class T>
struct is_stateless_lambda : std::integral_constant<bool, sizeof(T) == sizeof(std::true_type)>{};

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

static std::size_t index(const std::thread::id id) {
    static std::size_t nextindex = 0;
    static std::mutex my_mutex;
    static std::unordered_map<std::thread::id, std::size_t> ids;
    std::lock_guard<std::mutex> lock(my_mutex);
    auto iter = ids.find(id);
    if(iter == ids.end())
        return ids[id] = nextindex++;
    return iter->second;
}

template<typename Tfn, typename ... Args>
const string make_id(string id, const string& name, Tfn&& fn, Args&& ... args) {
	stringstream ss;
	if(!id.empty())
		id = "::" + id;

	ss << name << id << "-" << index(std::this_thread::get_id()) << " [" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << "] ";
	((ss << demangle(typeid(decltype(args)).name()) << "(" << int_to_hex((size_t)&args) << ") "), ...);
	return ss.str();
}

}


using namespace cv::v4d::detail;

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


    struct BranchState {
		string branchID_;
    	bool isEnabled_ = true;
    	bool isOnce_ = false;
    	bool isSingle_ = false;
    	bool condition = false;
    	bool isLocked_ = false;
    };

    int32_t workerIdx_ = -1;
    cv::Ptr<V4D> self_;
    cv::Ptr<Plan> plan_;
    const cv::Size initialSize_;

    int allocateFlags_;
    int configFlags_;
    int debugFlags_;

    cv::Rect viewport_;
    bool stretching_;
    int samples_;
    bool focused_ = false;
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
    cv::UMat captureFrame_;
    cv::UMat writerFrame_;
    cv::Point2f mousePos_;
    uint64_t seqNr_ = 0;
    bool showFPS_ = true;
    bool printFPS_ = false;
    bool showTracking_ = true;
    std::vector<std::pair<std::string, cv::Ptr<Transaction>>> transactions_;
    std::deque<BranchState> branchStateStack_;
    std::deque<std::pair<string, BranchType::Enum>> branchStack_;
    bool disableIO_ = false;
    std::string currentID_;
public:
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
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const string& title, int allocateFlags = AllocateFlags::DEFAULT, int configFlags = ConfigFlags::DEFAULT, int debugFlags = DebugFlags::DEFAULT, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, int allocateFlags = AllocateFlags::DEFAULT, int configFlags = ConfigFlags::DEFAULT, int debugFlags = DebugFlags::DEFAULT, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const V4D& v4d, const string& title);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();
    CV_EXPORTS const string getCurrentID() const;
    CV_EXPORTS cv::Ptr<V4D> setCurrentID(const string& p);
    CV_EXPORTS const int32_t& workerIndex() const;
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS cv::ogl::Texture2D& texture();
    CV_EXPORTS std::string title() const;

    struct Node {
    	string name_;
    	cv::Ptr<Transaction> tx_  = nullptr;
    	bool initialized() {
    		return tx_;
    	}
    };

    std::vector<cv::Ptr<Node>> nodes_;

    void makeGraph() {
    	CV_Assert(branchStack_.empty());
//    	cout << std::this_thread::get_id() << " ### MAKE PLAN ### " << endl;
    	for(const auto& t : transactions_) {
    		cv::Ptr<Node> n = new Node();
			n->name_ = t.first;
			n->tx_ = t.second;
			CV_Assert(!n->name_.empty());
			CV_Assert(n->tx_);
			nodes_.push_back(n);
    	}
    }


    void runGraph() {
		BranchType::Enum btype;
    	BranchState currentState;
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
						currentState.condition = n->tx_->enabled();
						if(currentState.isEnabled_) {
							currentState.isOnce_ = ((btype == BranchType::ONCE) || (btype == BranchType::PARALLEL_ONCE));
							currentState.isSingle_ = ((btype == BranchType::ONCE) || (btype == BranchType::SINGLE));
						} else {
							currentState.isOnce_ = false;
							currentState.isSingle_ = false;
							currentState.isEnabled_ = false;
						}

						if(currentState.isOnce_) {
							if((btype == BranchType::ONCE)) {
								currentState.condition = Global::once(n->name_) && currentState.condition;
							} else if((btype == BranchType::PARALLEL_ONCE)) {
								currentState.condition = !n->tx_->ran() && currentState.condition;
							} else {
								CV_Assert(false);
							}
						}

						currentState.isEnabled_ = currentState.isEnabled_ && currentState.condition;

						if(currentState.isEnabled_ && currentState.isSingle_) {
							CV_Assert(btype != BranchType::PARALLEL);

							if(Global::lock_node(currentState.branchID_)) {
//								cerr << "lock branch" << endl;
							}
							currentState.isLocked_ = true;
						}

						branchStateStack_.push_front(currentState);
					} else if(isElse) {
						if(branchStateStack_.empty())
							continue;
						currentState = branchStateStack_.front();
						currentState.isEnabled_ = !currentState.isEnabled_ && !currentState.condition;
						currentState.isOnce_ = false;
						currentState.condition = !currentState.condition;
						currentState.isSingle_ = false;

						if(currentState.isLocked_) {
							if(Global::try_unlock_node(currentState.branchID_)) {
//								cerr << "unlock else" << endl;
							}
						}

						currentState.isLocked_ = false;
						branchStateStack_.pop_front();
						branchStateStack_.push_front(currentState);
					} else if(isElseIf) {
						if(branchStateStack_.empty())
							continue;
						currentState = branchStateStack_.front();
						bool cond = n->tx_->enabled();
						currentState.isEnabled_ = !currentState.condition && !currentState.isEnabled_ && cond;
						currentState.condition = cond;
						currentState.isOnce_ = false;
						currentState.isSingle_ = false;
						//FIXME: missing locking on elseif
						if(currentState.isEnabled_) {
							currentState.isOnce_ = ((btype == BranchType::ONCE) || (btype == BranchType::PARALLEL_ONCE));
							currentState.isSingle_ = ((btype == BranchType::ONCE) || (btype == BranchType::SINGLE));
						}

						if(currentState.isLocked_) {
							if(Global::try_unlock_node(currentState.branchID_)) {
//								cerr << "unlock elseif" << endl;
							}
						}

						currentState.isLocked_ = false;
						branchStateStack_.pop_front();
						branchStateStack_.push_front(currentState);
					} else if(isEnd) {
						if(branchStateStack_.empty())
							continue;

						if(Global::try_unlock_node(branchStateStack_.front().branchID_)) {
//							cerr << "unlock end" << endl;
						}

						branchStateStack_.pop_front();
					} else {
						CV_Assert(false);
					}
				} else {
					CV_Assert(!n->tx_->isPredicate());
					currentState = !branchStateStack_.empty() ? branchStateStack_.front() : BranchState();
					if(currentState.isEnabled_) {
						auto lock = Global::try_get_node_lock(currentState.branchID_);
						if(lock)
						{
							std::lock_guard<std::mutex> guard(*lock.get());
							auto ctx = n->tx_->getContext();
							int res = ctx->execute(n->tx_->getViewport(), [n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [n,currentState](){
//									cerr << "locked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform();
								});
							});
							if(res > 0) {
								if(!this->disableIO_ && dynamic_pointer_cast<SourceContext>(ctx)) {
									this->setSequenceNumber(res);
								}
							} else {
								CV_LOG_WARNING(nullptr, "Context failed while: " + n->name_);
							}
						} else {
							auto ctx = n->tx_->getContext();
							int res = ctx->execute(n->tx_->getViewport(), [n,currentState]() {
								TimeTracker::getInstance()->execute(n->name_, [n,currentState](){
//									cerr << "unlocked: " << currentState.branchID_ << "->" << n->name_ << endl;
									n->tx_->perform();
								});
							});
							if(res > 0) {
								if(!this->disableIO_ && dynamic_pointer_cast<SourceContext>(ctx)) {
									this->setSequenceNumber(res);
								}
							} else {
								CV_LOG_WARNING(nullptr, "Context failed while: " + n->name_);
							}
						}
					}
				}
			}

			size_t lockCnt = Global::cound_node_locks();
//			cerr << "STATE STACK: " << branchStateStack_.size() << endl;
//			cerr << "LOCK STACK: " << lockCnt << endl;
			CV_Assert(branchStateStack_.empty());
			CV_Assert(lockCnt == 0);
			//FIXME unlock all on exception?
    	} catch(std::runtime_error& ex) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(Global::try_unlock_node(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw ex;
		} catch(std::exception& ex) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(Global::try_unlock_node(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw ex;
		} catch(...) {
			if(!branchStateStack_.empty() && branchStateStack_.front().isLocked_) {
				if(Global::try_unlock_node(currentState.branchID_)) {
//					cerr << "unlock exception" << endl;
				}
			}
			throw std::runtime_error("Unkown error.");
		}
	}

	void clearGraph() {
		branchStateStack_.clear();
		branchStack_.clear();
		transactions_.clear();
		nodes_.clear();
	}

    template<typename Tfn, typename ...Args>
    void add_transaction(cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
		auto tx = make_transaction(fn, std::forward<Args>(args)...);
		tx->setContext(ctx);
		tx->setBranchType(BranchType::NONE);
		tx->setViewport(getFramebufferViewport());
		transactions_.push_back({invocation, tx});
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(BranchType::Enum btype, cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
		auto tx = make_transaction(fn, std::forward<Args>(args)...);
		tx->setContext(ctx);
		tx->setBranchType(btype);
		tx->setViewport(getFramebufferViewport());
		transactions_.push_back({invocation, tx});
    }

    template <typename Tfn, typename ... Args>
    void init_context_call(Tfn fn, Args&& ... args) {
    	static_assert(detail::is_stateless_lambda<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value, "All passed functors must be stateless lambdas");
    	static_assert(std::conjunction<std::is_lvalue_reference<Args>...>::value, "All arguments must be l-value references");
    	(CV_UNUSED(args), ...);
    }


    template <typename Tfn, typename ... Args>
    typename std::enable_if<std::is_invocable_v<Tfn, Args...>, cv::Ptr<V4D>>::type
    gl(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "gl-1", fn, args...);
		std::function functor(fn);
		add_transaction(glCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> gl(int32_t idx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "gl" + std::to_string(idx), fn, args...);
		std::function<void((const int32_t&,Args...))> functor(fn);
		add_transaction<decltype(functor),const int32_t&>(glCtx(idx),id, std::forward<decltype(functor)>(functor), glCtx(idx)->getIndex(), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    typename std::enable_if<std::is_invocable_v<Tfn, Args...>, cv::Ptr<V4D>>::type
    ext(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "ext", fn, args...);
		std::function functor(fn);
		add_transaction(extCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> ext(int32_t idx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "ext" + std::to_string(idx), fn, args...);
		std::function<void((const int32_t&,Args...))> functor(fn);
		add_transaction<decltype(functor),const int32_t&>(extCtx(idx),id, std::forward<decltype(functor)>(functor), extCtx(idx)->getIndex(), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn>
    cv::Ptr<V4D> branch(Tfn fn) {
//        init_context_call(fn);
        const string id = make_id(getCurrentID(), "branch", fn);
        branchStack_.push_front({id, BranchType::PARALLEL});
		std::function functor = fn;
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> branch(Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "branch", fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
		std::function functor = fn;
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> branch(int workerIdx, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "branch-pin" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, BranchType::PARALLEL});
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};
		add_transaction(BranchType::PARALLEL, plainCtx(), id, wrap, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn>
    cv::Ptr<V4D> branch(BranchType::Enum type, Tfn fn) {
//        init_context_call(fn);
        const string id = make_id(getCurrentID(), "branch-type" + std::to_string((int)type) + "-", fn);
        branchStack_.push_front({id, type});
		std::function functor = fn;
		add_transaction(type, plainCtx(), id, functor);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> branch(BranchType::Enum type, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "branch-type" + std::to_string((int)type), fn, args...);
        branchStack_.push_front({id, type});
		std::function<bool(Args...)> functor = fn;
		add_transaction(type, plainCtx(), id, functor, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> branch(BranchType::Enum type, int workerIdx, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
        const string id = make_id(getCurrentID(), "branch-type-pin" + std::to_string((int)type) + "-" + std::to_string(workerIdx), fn, args...);
        branchStack_.push_front({id, type});
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};

		add_transaction(type, plainCtx(), id, wrap, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn>
    cv::Ptr<V4D> elseIfBranch(Tfn fn) {
//        init_context_call(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
		std::function functor = fn;
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> elseIfBranch(Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
		std::function functor = fn;
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> elseIfBranch(int workerIdx, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};
		add_transaction(BranchType::PARALLEL, plainCtx(), id, wrap, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn>
    cv::Ptr<V4D> elseIfBranch(BranchType::Enum type, Tfn fn) {
//        init_context_call(fn);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
		std::function functor = fn;
		add_transaction(type, plainCtx(), id, functor);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> elseIfBranch(BranchType::Enum type, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
		std::function<bool(Args...)> functor = fn;
		add_transaction(type, plainCtx(), id, functor, std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> elseIfBranch(BranchType::Enum type, int workerIdx, Tfn fn, Args&& ... args) {
//        init_context_call(fn, args...);
    	auto current = branchStack_.front();
    	string id = "[elseif]" + current.first;
        std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};

		add_transaction(type, plainCtx(), id, wrap, std::forward<Args>(args)...);
		return self();
    }

	cv::Ptr<V4D> endBranch() {
    	auto current = branchStack_.front();
    	branchStack_.pop_front();
        string id = "[end]" + current.first;
		std::function functor = [](){ return true; };
		add_transaction(current.second, plainCtx(), id, functor);
		return self();
    }

    cv::Ptr<V4D> elseBranch() {
    	auto current = branchStack_.front();
    	string id = "[else]" + current.first;
		std::function functor = [](){ return true; };
		add_transaction(current.second, plainCtx(), id, functor);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> fb(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getCurrentID(), "fb", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;
		using Tfbbase = typename std::remove_cv<Tfb>::type;

		static_assert((std::is_same<Tfb, cv::UMat&>::value || std::is_same<Tfb, const cv::UMat&>::value) || !"The first argument must be eiter of type 'cv::UMat&' or 'const cv::UMat&'");
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(fbCtx(),id, std::forward<decltype(functor)>(functor), std::forward<Tfb>(fbCtx()->view()), std::forward<Args>(args)...);
		return self();
    }

    cv::Ptr<V4D> clear(cv::Scalar bgra = cv::Scalar(0.0, 0.0, 0.0, 1.0)) {
        gl([](const cv::Scalar& clearColor) {
        	glClearColor(clearColor[0] / 255.0, clearColor[1] / 255.0, clearColor[2] / 255.0, clearColor[3] / 255.0);
        	//FIXME also clear depth and stencil?
        	glClear(GL_COLOR_BUFFER_BIT);
        }, bgra);
    	return self();
    }

    cv::Ptr<V4D> capture() {
    	if(disableIO_)
    		return self();
    	capture([](const cv::UMat& inputFrame, cv::UMat& f){
    		if(!inputFrame.empty())
    			inputFrame.copyTo(f);
    	}, captureFrame_);

        fb([](cv::UMat& framebuffer, const cv::UMat& f) {
        	if(!f.empty()) {
        		if(f.size() != framebuffer.size())
        			resizePreserveAspectRatio(f, framebuffer, framebuffer.size());
        		else
        			f.copyTo(framebuffer);
        	}
        }, captureFrame_);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> capture(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

    	if(disableIO_)
    		return self();
        const string id = make_id(getCurrentID(), "capture", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,const cv::UMat&>::value) || !"The first argument must be of type 'const cv::UMat&'");
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sourceCtx()),id, std::forward<decltype(functor)>(functor), sourceCtx()->sourceBuffer(), std::forward<Args>(args)...);
		return self();
    }

    cv::Ptr<V4D> write() {
    	if(disableIO_)
    		return self();

        fb([](const cv::UMat& framebuffer, cv::UMat& f) {
            framebuffer.copyTo(f);
        }, writerFrame_);

    	write([](cv::UMat& outputFrame, const cv::UMat& f){
   			f.copyTo(outputFrame);
    	}, writerFrame_);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> write(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);


    	if(disableIO_)
    		return self();
        const string id = make_id(getCurrentID(), "write", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,cv::UMat&>::value) || !"The first argument must be of type 'cv::UMat&'");
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sinkCtx()),id, std::forward<decltype(functor)>(functor), sinkCtx()->sinkBuffer(), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> nvg(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id(getCurrentID(), "nvg", fn, args...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(nvgCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> bgfx(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id(getCurrentID(), "bgfx", fn, args...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(bgfxCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
		return self();
    }

    template <typename Tfn, typename ... Args>
    cv::Ptr<V4D> plain(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id(getCurrentID(), "plain", fn, args...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(fbCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
		return self();
    }

    template<typename Tfn, typename ... Args>
    void imgui(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);

        if(!hasImguiCtx())
        	return;

        auto s = self();

        imguiCtx()->build([s, fn, &args...]() {
			fn(s, args...);
		});
    }
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

    template<typename Tplan, typename ... Args>
	void run(cv::Ptr<Tplan> plan, int32_t workers, Args ... args) {
		plan_ = std::static_pointer_cast<Plan>(plan);

		//the first sequence number is 1!
		static Resequence reseq(1);
		//for now, if automatic determination of the number of workers is requested,
		//set workers always to 2
		CV_Assert(workers > -2);
		if(workers == -1) {
			workers = 2;
		} else {
			++workers;
		}

		static cv::utils::logging::LogLevel initial_level;
		std::vector<std::thread*> threads;
		{
			static std::mutex runMtx;
			std::lock_guard<std::mutex> lock(runMtx);
			cv::setNumThreads(0);

			if(Global::is_first_run()) {
				Global::set_main_id(std::this_thread::get_id());
				CV_LOG_INFO(nullptr, "Starting with " << workers << " workers");
				initial_level = cv::utils::logging::getLogLevel();
			}


			if(Global::is_main()) {
				cv::Size sz = this->initialSize();
				const string title = this->title();
				auto src = this->getSource();
				auto sink = this->getSink();
				Global::set<size_t>(Global::WORKERS_STARTED, workers);
				Global::set<LatchPtr>(Global::WORKER_READY_BARRIER, cv::makePtr<Latch>(workers + 1));
				std::vector<cv::Ptr<Tplan>> plans;
				//make sure all Plans are constructed before starting the workers
				for (size_t i = 0; i < workers; ++i) {
					plans.push_back(new Tplan(plan->viewport(), args...));
				}

				for (size_t i = 0; i < workers; ++i) {
					threads.push_back(
						new std::thread(
							[this, i, workers, src, sink, plans, args...] {
								CV_LOG_ONCE_WARNING(nullptr, "Temporary setting log level to warning for workers.");
								cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

								CV_LOG_DEBUG(nullptr, "Creating worker: " << i);
								cv::Ptr<cv::v4d::V4D> worker = V4D::make(*this, this->title() + "-worker-" + std::to_string(i));
								if (src) {
									worker->setSource(src);
								}
								if (sink) {
									worker->setSink(sink);
								}
								cv::Ptr<Tplan> newPlan = plans[i];

								worker->setCurrentID(newPlan->id());
								worker->run(newPlan, 0, args...);
							}
						)
					);
				}
			} else {
//			    cerr << "Setting worker thread niceness from: " << getpriority(PRIO_PROCESS, gettid()) << " to: " << 1 << endl;
//
//			    if (setpriority(PRIO_PROCESS, gettid(), 1))
//			        std::cout << "Failed to setpriority: " << std::strerror(errno) << '\n';
			}
		}

		CLExecScope_t scope(this->fbCtx()->getCLExecContext());
		this->fbCtx()->makeCurrent();

		if(Global::is_main()) {
			this->printSystemInfo();
		} else {
			try {
				CV_LOG_DEBUG(nullptr, "Setup on worker: " << workerIndex());
				plan->setup(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Setup failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(nullptr, "Setup finished: " << workerIndex());
		}
		if(Global::is_main()) {
			try {
				CV_LOG_DEBUG(nullptr, "Loading GUI");
				setCurrentID(plan->id());
				plan->gui(self());
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Loading GUI failed: %s", ex.what()));
			}
		} else {

			try {
				CV_LOG_DEBUG(nullptr, "Main inference on worker: " << workerIndex());
				plan->infer(self());
				this->makeGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("Main inference failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(nullptr, "Main inference finished: " << workerIndex());
		}

		static std::binary_semaphore frame_sync_render(0);
		static std::binary_semaphore frame_sync_sema_swap(0);
		try {
			if(Global::is_main()) {
				CV_LOG_INFO(nullptr, "Display thread waiting for workers to get ready.");
				Global::get<LatchPtr>(Global::WORKER_READY_BARRIER)->arrive_and_wait();
				CV_LOG_INFO(nullptr, "Display thread started.");
				while(keepRunning()) {
					if(configFlags() & ConfigFlags::DISPLAY_MODE) {
						if(!this->display()) {
							frame_sync_render.release();
							break;
						}
						frame_sync_render.release();
						//refresh-rate depends on swap interval (1) for sync
						frame_sync_sema_swap.acquire();
					} else {
						if(!this->display()) {
							break;
						}
					}
				}
			} else {
				static std::once_flag oflag;
				CV_LOG_DEBUG(nullptr, "Worker " << workerIndex() << " waiting for the other workers to get ready.");
				Global::get<LatchPtr>(Global::WORKER_READY_BARRIER)->arrive_and_wait();
				std::call_once(oflag, [this](){
					CV_LOG_WARNING(nullptr, "Re-enabling logging.");
					CV_LOG_INFO(nullptr, "Starting pipelines with " << this->nodes_.size() << " nodes and " << Global::workers_started() << " workers.");
				});
				cv::utils::logging::setLogLevel(initial_level);

				while(keepRunning()) {
					event::poll();
					if(!hasSource() || (hasSource() && !getSource()->isOpen())) {
						Global::on<size_t>(Global::RUN_COUNT, [this](size_t& s) {
							this->setSequenceNumber(++s);
							return s;
						});
					}

					if(configFlags() & ConfigFlags::DISPLAY_MODE) {
						frame_sync_sema_swap.release();
						this->runGraph();
						size_t seq = this->getSequenceNumber();
						reseq.waitFor(seq, [](uint64_t s) {
							frame_sync_render.acquire();
						});

						if(!this->display()) {
							frame_sync_sema_swap.release();
							break;
						}
					} else {
						this->runGraph();
						bool result = false;
						reseq.waitFor(this->getSequenceNumber(), [&result, this](uint64_t s) {
							CV_UNUSED(s);
							result = this->display();
						});

						if(!result)
							break;
					}
				};
			}
		} catch(std::runtime_error& ex) {
			CV_LOG_WARNING(nullptr, "Pipeline terminated: " << ex.what());
		} catch(std::exception& ex) {
			CV_LOG_WARNING(nullptr, "Pipeline terminated: " << ex.what());
		} catch(...) {
			CV_LOG_WARNING(nullptr, "Pipeline terminated with unknown error.");
		}
		requestFinish();
		reseq.finish();
		if(configFlags() & ConfigFlags::DISPLAY_MODE) {
			if(Global::is_main()) {
				for(size_t i = 0; i < Global::workers_started(); ++i)
					frame_sync_render.release();
			} else {
				frame_sync_sema_swap.release();
			}
    	}
		if(!Global::is_main()) {
			this->clearGraph();
			CV_LOG_DEBUG(nullptr, "Starting teardown on worker: " << workerIndex());
			try {
				plan->teardown(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline tear-down failed: %s", ex.what()));
			}
			CV_LOG_DEBUG(nullptr, "Teardown complete on worker: " << workerIndex());
		} else {
			for(auto& t : threads)
				t->join();
			CV_LOG_INFO(nullptr, "All threads terminated.");
		}
	}
/*!
     * Called to feed an image directly to the framebuffer
     */
    CV_EXPORTS void feed(cv::UMat& in);

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
     * Get the current viewport reference.
     * @return The current viewport reference.
     */
    CV_EXPORTS cv::Rect& viewport();

    CV_EXPORTS cv::Rect getFramebufferViewport();
    CV_EXPORTS cv::Ptr<V4D> setFramebufferViewport(const cv::Rect& vp);

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
    CV_EXPORTS const cv::Size& initialSize() const;
    CV_EXPORTS const cv::Size& fbSize() const;
    /*!
     * Set the window size
     * @param sz The future size of the window.
     */
    CV_EXPORTS void setSize(const cv::Size& sz);
    /*!
     * Get the window size.
     * @return The window size.
     */
    CV_EXPORTS const cv::Size size();
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
    CV_EXPORTS cv::Ptr<V4D> setDisableIO(bool d);

    CV_EXPORTS bool isFullscreen();
    /*!
     * Enable or disable fullscreen mode.
     * @param f if true enable fullscreen mode else disable.
     */
    CV_EXPORTS void setFullscreen(bool f);
    /*!
     * Determines if the window is resizeable.
     * @return true if the window is resizeable.
     */
    CV_EXPORTS bool isResizable();
    /*!
     * Set the window resizable.
     * @param r if r is true set the window resizable.
     */
    CV_EXPORTS void setResizable(bool r);
    /*!
     * Determine if the window is visible.
     * @return true if the window is visible.
     */
    CV_EXPORTS bool isVisible();
    /*!
     * Set the window visible or invisible.
     * @param v if v is true set the window visible.
     */
    CV_EXPORTS void setVisible(bool v);
    /*!
     * Enable/Disable scaling the framebuffer during blitting.
     * @param s if true enable scaling.
     */
    CV_EXPORTS void setStretching(bool s);
    /*!
     * Determine if framebuffer is scaled during blitting.
     * @return true if framebuffer is scaled during blitting.
     */
    CV_EXPORTS bool isStretching();
    /*!
     * Determine if the window is closed.
     * @return true if the window is closed.
     */
    CV_EXPORTS bool isClosed();
    /*!
     * Close the window.
     */
    CV_EXPORTS void close();
    /*!
     * Print basic system information to stderr.
     */
    CV_EXPORTS void printSystemInfo();
    CV_EXPORTS int allocateFlags();
    CV_EXPORTS int configFlags();
    CV_EXPORTS int debugFlags();
private:
    V4D(const V4D& v4d, const string& title);
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, int allocFlags, int confFlags, int debFlags, int samples);

    void swapContextBuffers();
    bool display();
protected:
    cv::Ptr<V4D> self();

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
    size_t numGlCtx();
    size_t numExtCtx();

    cv::Ptr<Plan> plan();
    GLFWwindow* getGLFWWindow() const;
    bool isFocused();
    void setFocused(bool f);
    void setSequenceNumber(size_t seq);
    uint64_t getSequenceNumber();
};
}
} /* namespace cv */

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */

