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
#include "detail/resequence.hpp"
#include "events.hpp"

#include <type_traits>
#include <shared_mutex>
#include <iostream>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "imgui.h"

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

enum class AllocateFlags {
    NONE = 0,
    NANOVG = 1,
    IMGUI = 2,
    ALL = NANOVG | IMGUI
};

enum class BranchType {
	NONE = 0,
	SINGLE = 1,
	PARALLEL = 2,
	ONCE = 4,
	PARALLEL_ONCE = 8,
};

class Plan {
	const cv::Size sz_;
	const cv::Rect vp_;
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

static std::size_t index(const std::thread::id id)
{
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
const string make_id(const string& prefix, const string& name, Tfn&& fn, Args&& ... args) {
	stringstream ss;
	ss << "(" << prefix << ") " << name << " (" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << ")-" << index(std::this_thread::get_id());
	((ss << ',' << demangle(typeid(decltype(args)).name()) << int_to_hex((size_t)&args)), ...);
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
    friend class Source;
    friend class Sink;

    int32_t workerIdx_ = -1;
    cv::Ptr<V4D> self_;
    cv::Ptr<Plan> plan_;
    const cv::Size initialSize_;
    AllocateFlags flags_;
    bool debug_;
    cv::Rect viewport_;
    bool stretching_;
    int samples_;
    bool focused_ = false;
    cv::Ptr<FrameBufferContext> mainFbContext_ = nullptr;
    cv::Ptr<SourceContext> sourceContext_ = nullptr;
    cv::Ptr<SinkContext> sinkContext_ = nullptr;
    cv::Ptr<NanoVGContext> nvgContext_ = nullptr;
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
    std::vector<std::tuple<std::string,bool,long>> accesses_;
    std::map<std::string, cv::Ptr<Transaction>> transactions_;
    bool disableIO_ = false;
    std::string prefix_;
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
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const string& title, AllocateFlags flags = AllocateFlags::ALL, bool offscreen = false, bool debug = false, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags = AllocateFlags::ALL, bool offscreen = false, bool debug = false, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const V4D& v4d, const string& title);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();
    CV_EXPORTS const string getPrefix() const;
    CV_EXPORTS void setPrefix(const string& p);
    CV_EXPORTS const int32_t& workerIndex() const;
    CV_EXPORTS size_t workers_running();
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS cv::ogl::Texture2D& texture();
    CV_EXPORTS std::string title() const;

    struct Node {
    	string name_;
    	std::set<long> read_deps_;
    	std::set<long> write_deps_;
    	cv::Ptr<Transaction> tx_  = nullptr;
    	bool initialized() {
    		return tx_;
    	}
    };

    std::vector<cv::Ptr<Node>> nodes_;

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
    		const long& dep = std::get<2>(t);
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

    struct BranchState {
		string branchID;
    	bool isEnabled_ = true;
    	bool isOnce_ = false;
    };

    std::deque<BranchState> branchStack_;

    void runGraph() {
		bool locker = false;
		bool openBranch = false;
		BranchType btype;

    	BranchState currentState;
    	try {
			for (auto& n : nodes_) {
				btype = n->tx_->getBranchType();

				if(btype != BranchType::NONE) {
					if(!openBranch) {
						openBranch = true;
						currentState.branchID = n->name_;
						currentState.isOnce_ = ((btype == BranchType::ONCE) || (btype == BranchType::PARALLEL_ONCE));
						currentState.isEnabled_ = n->tx_->enabled();
						if(currentState.isEnabled_) {
							if(((btype == BranchType::ONCE) || (btype == BranchType::SINGLE))) {
								CV_Assert(btype != BranchType::PARALLEL);
								CV_Assert(!Global::isLocking());
								Global::mutex().lock();
								locker = true;
								Global::setLocking(true);
							}

							if(currentState.isOnce_) {
								if((btype == BranchType::ONCE)) {
									currentState.isEnabled_ = Global::once(n->name_);
								} else if((btype == BranchType::PARALLEL_ONCE)) {
									currentState.isEnabled_ = !n->tx_->ran();
								} else {
									CV_Assert(false);
								}
							}
						}
						branchStack_.push_front(currentState);

					} else {
						CV_Assert(currentState.isOnce_ == ((btype == BranchType::ONCE) || (btype == BranchType::PARALLEL_ONCE)));
						openBranch = false;

						if(locker) {
							CV_Assert(Global::isLocking());
							locker = false;
							Global::setLocking(false);
							Global::mutex().unlock();
						}

						CV_Assert(!branchStack_.empty());
						branchStack_.pop_front();
						if(!branchStack_.empty())
							currentState = branchStack_.front();
						else
							currentState = BranchState();
						CV_Assert(currentState.isEnabled_);
					}
				} else {
					CV_Assert(!n->tx_->isPredicate());

					if (currentState.isEnabled_) {
						if(!locker && Global::isLocking()) {
							std::lock_guard<std::mutex>(Global::mutex());
							n->tx_->getContext()->execute([n]() {
								TimeTracker::getInstance()->execute(n->name_, [n](){
									cerr << "locked: " << n->name_ << endl;
									n->tx_->perform();
								});
							});
						} else {
							n->tx_->getContext()->execute([n]() {
								TimeTracker::getInstance()->execute(n->name_, [n](){
									cerr << "unlocked: " << n->name_ << endl;
									n->tx_->perform();
								});
							});
						}
					}
				}
			}
		} catch(std::exception& ex) {
			if(locker) {
				CV_Assert(Global::isLocking());
				locker = false;
				Global::setLocking(false);
				Global::mutex().unlock();
			}
			throw ex;
		} catch(...) {
			if(locker) {
				CV_Assert(Global::isLocking());
				Global::mutex().unlock();
			}
			CV_Error(cv::Error::StsError, "Unknown error in pipeline");
		}
	}

	void clearGraph() {
		nodes_.clear();
		accesses_.clear();
	}

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::false_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
    	//disabled
    }

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::true_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
//    	cout << "access: " << std::this_thread::get_id() << " " << context << string(read ? " <- " : " -> ") << demangle(typeid(std::remove_const_t<T>).name()) << "(" << (long)tp << ") " << endl;
    	accesses_.push_back(std::make_tuple(context, read, (long)tp));
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
    	auto it = transactions_.find(invocation);
    	if(it == transactions_.end()) {
    		auto tx = make_transaction(fn, std::forward<Args>(args)...);
    		tx->setContext(ctx);
    		transactions_.insert({invocation, tx});
    	}
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(BranchType btype, cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
    	auto it = transactions_.find(invocation);
    	if(it == transactions_.end()) {
    		auto tx = make_transaction(fn, std::forward<Args>(args)...);
    		tx->setContext(ctx);
    		tx->setBranchType(btype);
    		transactions_.insert({invocation, tx});
    	}
    }

    template <typename Tfn, typename ... Args>
    void init_context_call(Tfn fn, Args&& ... args) {
    	static_assert(detail::is_stateless_lambda<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value, "All passed functors must be stateless lambdas");
    	static_assert(std::conjunction<std::is_lvalue_reference<Args>...>::value, "All arguments must be l-value references");
    }


    template <typename Tfn, typename ... Args>
    typename std::enable_if<std::is_invocable_v<Tfn, Args...>, void>::type
    gl(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);
        const string id = make_id(getPrefix(), "gl-1", fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function functor(fn);
		add_transaction(glCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void gl(int32_t idx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "gl" + std::to_string(idx), fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function<void((const int32_t&,Args...))> functor(fn);
		add_transaction<decltype(functor),const int32_t&>(glCtx(idx),id, std::forward<decltype(functor)>(functor), glCtx(idx)->getIndex(), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    typename std::enable_if<std::is_invocable_v<Tfn, Args...>, void>::type
    ext(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);
        const string id = make_id(getPrefix(), "ext", fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function functor(fn);
		add_transaction(extCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void ext(int32_t idx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "ext" + std::to_string(idx), fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function<void((const int32_t&,Args...))> functor(fn);
		add_transaction<decltype(functor),const int32_t&>(extCtx(idx),id, std::forward<decltype(functor)>(functor), extCtx(idx)->getIndex(), std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void branch(Tfn fn) {
        init_context_call(fn);
        const string id = make_id(getPrefix(), "branch", fn);
		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void branch(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "branch", fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor = fn;
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void branch(int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "branch-pin" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};
		add_transaction(BranchType::PARALLEL, plainCtx(), id, wrap, std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void branch(BranchType type, Tfn fn) {
        init_context_call(fn);
        const string id = make_id(getPrefix(), "branch-type" + std::to_string((int)type) + "-", fn);
		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(type, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void branch(BranchType type, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "branch-type" + std::to_string((int)type), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = fn;
		add_transaction(type, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void branch(BranchType type, int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "branch-type-pin" + std::to_string((int)type) + "-" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};

		add_transaction(type, plainCtx(), id, wrap, std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void endbranch(Tfn fn) {
        init_context_call(fn);
        const string id = make_id(getPrefix(), "endbranch", fn);

		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "endbranch", fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this](Args&& ... args){
			return true;
		};
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "endbranch-pin" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this, workerIdx](Args&& ... args){
			return this->workerIndex() == workerIdx;
		};
		add_transaction(BranchType::PARALLEL, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void endbranch(BranchType type, Tfn fn) {
        init_context_call(fn);
        const string id = make_id(getPrefix(), "endbranch-type" + std::to_string((int)type) + "-", fn);

		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(type, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(BranchType type, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "endbranch-type" + std::to_string((int)type) + "-", fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this](Args&& ... args){
			return true;
		};
		add_transaction(type, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(BranchType type, int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "endbranch-pin-type" + std::to_string((int)type) + "-" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this, workerIdx](Args&& ... args){
			return this->workerIndex() == workerIdx;
		};
		add_transaction(type, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void fb(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "fb", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;
		using Tfbbase = typename std::remove_cv<Tfb>::type;

		static_assert((std::is_same<Tfb, cv::UMat&>::value || std::is_same<Tfb, const cv::UMat&>::value) || !"The first argument must be eiter of type 'cv::UMat&' or 'const cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<static_not<typename std::is_const<Tfbbase>::type>, cv::UMat, Tfb, Args...>(id, false, &fbCtx()->fb());
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(fbCtx(),id, std::forward<decltype(functor)>(functor), fbCtx()->fb(), std::forward<Args>(args)...);
    }

    void capture() {
    	if(disableIO_)
    		return;
    	capture([](const cv::UMat& inputFrame, cv::UMat& f){
    		if(!inputFrame.empty())
    			inputFrame.copyTo(f);
    	}, captureFrame_);

        fb([](cv::UMat& frameBuffer, const cv::UMat& f) {
        	if(!f.empty())
        		f.copyTo(frameBuffer);
        }, captureFrame_);
    }

    template <typename Tfn, typename ... Args>
    void capture(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);


    	if(disableIO_)
    		return;
        const string id = make_id(getPrefix(), "capture", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,const cv::UMat&>::value) || !"The first argument must be of type 'const cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sourceCtx()->sourceBuffer());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sourceCtx()),id, std::forward<decltype(functor)>(functor), sourceCtx()->sourceBuffer(), std::forward<Args>(args)...);
    }

    void write() {
    	if(disableIO_)
    		return;

        fb([](const cv::UMat& frameBuffer, cv::UMat& f) {
            frameBuffer.copyTo(f);
        }, writerFrame_);

    	write([](cv::UMat& outputFrame, const cv::UMat& f){
    		f.copyTo(outputFrame);
    	}, writerFrame_);
    }

    template <typename Tfn, typename ... Args>
    void write(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);


    	if(disableIO_)
    		return;
        const string id = make_id(getPrefix(), "write", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,cv::UMat&>::value) || !"The first argument must be of type 'cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sinkCtx()->sinkBuffer());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, false, &sinkCtx()->sinkBuffer());
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sinkCtx()),id, std::forward<decltype(functor)>(functor), sinkCtx()->sinkBuffer(), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void nvg(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "nvg", fn, args...);
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(nvgCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void plain(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id(getPrefix(), "plain", fn, args...);
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(fbCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
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

		static Resequence reseq;
		//for now, if automatic determination of the number of workers is requested,
		//set workers always to 2
		CV_Assert(workers > -2);
		if(workers == -1) {
			workers = 2;
		} else {
			++workers;
		}

		std::vector<std::thread*> threads;
		{
			static std::mutex runMtx;
			std::unique_lock<std::mutex> lock(runMtx);

			cerr << "run plan: " << std::this_thread::get_id() << " workers: " << workers << endl;

			if(Global::is_first_run()) {
				Global::set_main_id(std::this_thread::get_id());
				cerr << "Starting with " << workers - 1<< " extra workers" << endl;
				cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
			}

			if(workers > 1) {
				cv::setNumThreads(0);
			}

			if(Global::is_main()) {
				cv::Size sz = this->initialSize();
				const string title = this->title();
				bool debug = this->debug_;
				auto src = this->getSource();
				auto sink = this->getSink();
				Global::set_workers_started(workers);
				std::vector<cv::Ptr<Tplan>> plans;
				//make sure all Plans are constructed before starting the workers
				for (size_t i = 0; i < workers; ++i) {
					plans.push_back(new Tplan(plan->viewport(), args...));
				}
				for (size_t i = 0; i < workers; ++i) {
					threads.push_back(
						new std::thread(
							[this, i, src, sink, plans, args...] {
								cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
								cv::Ptr<cv::v4d::V4D> worker = V4D::make(*this, this->title() + "-worker-" + std::to_string(i));
								if (src) {
									worker->setSource(src);
								}
								if (sink) {
									worker->setSink(sink);
								}
								cv::Ptr<Tplan> newPlan = plans[i];
								worker->run(newPlan, 0, args...);
							}
						)
					);
				}
			}
		}

		CLExecScope_t scope(this->fbCtx()->getCLExecContext());
		this->fbCtx()->makeCurrent();

		if(Global::is_main()) {
			this->printSystemInfo();
		} else {
			try {
				plan->setup(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
				if(!Global::is_main() && Global::workers_started() == Global::next_worker_ready()) {
					cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
				}
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline setup failed: %s", ex.what()));
			}
		}
		if(Global::is_main()) {
			try {
				plan->gui(self());
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("GUI setup failed: %s", ex.what()));
			}
		} else {
			plan->infer(self());
			this->makeGraph();
		}

		try {
			std::mutex pollMtx;
			if(Global::is_main()) {
				do {
					//refresh-rate depends on swap interval (1) for sync
				} while(keepRunning() && this->display());
				requestFinish();
				reseq.finish();
			} else {
				cerr << "Starting pipeling with " << this->nodes_.size() << " nodes." << endl;

				static std::mutex seqMtx;
				do {
					reseq.notify();
					uint64_t seq;
					{
						std::lock_guard<std::mutex> lock(seqMtx);
						seq = Global::next_run_cnt();
					}

					this->runGraph();
					reseq.waitFor(seq);
					{
						std::lock_guard<std::mutex> lock(pollMtx);
						event::poll();

					}
				} while(keepRunning() && this->display());
			}
		} catch(std::exception& ex) {
			requestFinish();
			reseq.finish();
			CV_LOG_WARNING(nullptr, "-> pipeline terminated: " << ex.what());
		}

		if(!Global::is_main()) {
			this->clearGraph();

			try {
				plan->teardown(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline tear-down failed: %s", ex.what()));
			}
		} else {
			for(auto& t : threads)
				t->join();
		}
	}
/*!
     * Called to feed an image directly to the framebuffer
     */
	void feed(cv::UMat& in);
    /*!
     * Fetches a copy of frambuffer
     * @return a copy of the framebuffer
     */
    CV_EXPORTS cv::UMat fetch();

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
    CV_EXPORTS void setFramebufferViewport(const cv::Rect& vp);

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
    CV_EXPORTS cv::Size size();
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
    CV_EXPORTS void setDisableIO(bool d);

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
private:
    V4D(const V4D& v4d, const string& title);
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples);

    void swapContextBuffers();
    bool display();
protected:
    AllocateFlags flags();
    cv::Ptr<V4D> self();

    cv::Ptr<FrameBufferContext> fbCtx() const;
    cv::Ptr<SourceContext> sourceCtx();
    cv::Ptr<SinkContext> sinkCtx();
    cv::Ptr<NanoVGContext> nvgCtx();
    cv::Ptr<PlainContext> plainCtx();
    cv::Ptr<ImGuiContextImpl> imguiCtx();
    cv::Ptr<GLContext> glCtx(int32_t idx = 0);
    cv::Ptr<ExtContext> extCtx(int32_t idx = 0);

    bool hasFbCtx();
    bool hasSourceCtx();
    bool hasSinkCtx();
    bool hasNvgCtx();
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
    const uint64_t& sequenceNumber();
};
}
} /* namespace cv */

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */

