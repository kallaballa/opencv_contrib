// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_UTIL_HPP_
#define SRC_OPENCV_V4D_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"
#include "detail/framebuffercontext.hpp"
#include <filesystem>
#include <string>
#include <iostream>

#include "threadsafeanymap.hpp"
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <unistd.h>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>
#include <thread>
#include <latch>
#include <deque>

using std::cout;
using std::endl;

namespace cv {
namespace v4d {
namespace detail {

template <typename, typename = void>
struct has_call_operator_t : std::false_type {};

template <typename T>
struct has_call_operator_t<T, std::void_t<decltype(&T::operator())>> : std::is_same<std::true_type, std::true_type>
{};

template <typename, typename = void>
struct has_return_type_t : std::false_type {};

template <typename T>
struct has_return_type_t<T, std::void_t<decltype(&T::return_type)>> : std::is_same<std::true_type, std::true_type>
{};

template<typename T>
struct is_callable : public std::conjunction<has_call_operator_t<T>, has_return_type_t<T>> {
};

//https://stackoverflow.com/a/27885283/1884837
template<class T>
struct function_traits : function_traits<decltype(&T::operator())> {
};

// partial specialization for function type
template<class R, class... Args>
struct function_traits<R(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for function pointer
template<class R, class... Args>
struct function_traits<R (*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for std::function
template<class R, class... Args>
struct function_traits<std::function<R(Args...)>> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for pointer-to-member-function (i.e., operator()'s)
template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...) const> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};


//https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
CV_EXPORTS std::string demangle(const char* name);

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
    typedef std::function<_Res(_ArgTypes...)> function_type;

    static void bind(function_type&& f)
    { instance().fn_.swap(f); }

    static void bind(const function_type& f)
    { instance().fn_=f; }

    static _Res invoke(_ArgTypes... args)
    { return instance().fn_(args...); }

    typedef decltype(&fun_ptr_helper::invoke) pointer_type;
    static pointer_type ptr()
    { return &invoke; }

private:
    static fun_ptr_helper& instance()
    {
        static fun_ptr_helper inst_;
        return inst_;
    }

    fun_ptr_helper() {}

    function_type fn_;
};

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr(const std::function<_Res(_ArgTypes...)>& f)
{
    fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind(f);
    return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function(T *t)
{
    return {t};
}

//https://stackoverflow.com/a/33047781/1884837
class Lambda {
    template<typename T>
    static const void* fn(const void* new_fn = nullptr) {
        CV_Assert(new_fn);
    	return new_fn;
    }
	template<typename Tret, typename T>
    static Tret lambda_ptr_exec() {
        return (Tret) (*(T*)fn<T>());
    }
public:
	//FIXME race condition?
    template<typename Tret = void, typename Tfp = Tret(*)(), typename T>
    static Tfp ptr(T& t) {
        fn<T>(&t);
        return (Tfp) lambda_ptr_exec<Tret, T>;
    }
};

CV_EXPORTS size_t cnz(const cv::UMat& m);

}
using std::string;
class V4D;
class Plan;

inline uint64_t get_epoch_nanos() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

class SharedVariables {
	std::mutex sharedVarsMtx_;
	std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>> sharedVars_;
	typedef typename std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>>::iterator SharedVarsIter;

	template<typename T>
	void check(const T& shared) {
		off_t varOffset = reinterpret_cast<size_t>(&shared);
		off_t registeredOffset = 0;
		off_t registeredSize = 0;
		for(const auto& p : sharedVars_) {
			registeredOffset = p.first;
			if(varOffset > registeredOffset) {
				registeredSize = p.second.first;
				if(varOffset < (registeredOffset + registeredSize)) {
					throw std::runtime_error("You are trying to register a member of shared variable as shared variable");
				}
			}
		}
	}

public:
	template<typename T>
	bool isShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		return sharedVars_.find(reinterpret_cast<size_t>(&shared)) != sharedVars_.end();
	}

	template<typename T, bool Tcheck = true>
	void registerShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		auto it = sharedVars_.find(reinterpret_cast<size_t>(&shared));
		if(it == sharedVars_.end()) {
			if constexpr(Tcheck) {
				check(shared);
			}

			sharedVars_.insert({reinterpret_cast<size_t>(&shared), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
		}
	}

	template<typename T>
	void safe_copy(const T& from, T& to) {
		std::mutex* mtx = getMutexPtr(from, false);
		mtx = mtx == nullptr ? getMutexPtr(to, false) : mtx;
		if(mtx == nullptr)
			throw std::runtime_error("Internal error: Trying to safe copy non-shared variables.");

		std::lock_guard<std::mutex> guard(*mtx);
		to = copy(from);
	}

	template<typename T>
	static void copy(const T& from, T& to) {
			to = copy_construct(from);
	}

	static void copy(const cv::UMat& from, cv::UMat& to) {
		if(from.empty())
			return;
		if(to.empty())
			to.create(from.size(), from.type());
		from.copyTo(to.getMat(cv::ACCESS_WRITE));
	}

	template<typename T>
	static T safe_copy(const T& from) {
		T to;
		safe_copy(from, to);
		return to;
	}

	template<typename T>
	static T copy(const T& from) {
		T to;
		copy(from, to);
		return to;
	}

	template<typename T>
	static T copy_construct(const T& t) {
		return t;
	}


	template<typename T>
	std::mutex* getMutexPtr(const T& shared, bool check = true) {
		SharedVarsIter it, end;
		cv::Ptr<std::mutex> mtx = nullptr;
			std::lock_guard<std::mutex> guard(sharedVarsMtx_);
			it = sharedVars_.find(reinterpret_cast<size_t>(&shared));
			end = sharedVars_.end();
			if(it != end) {
				mtx = (*it).second.second;
			}

			if(check && !mtx)
				throw std::runtime_error("You are trying to lock a non-shared variable");

			return mtx.get();
	}

	template<typename T>
	void lock(const T& shared) {
		getMutexPtr(shared)->lock();
	}

	template<typename T>
	void unlock(const T& shared) {
		getMutexPtr(shared)->unlock();
	}

	template<typename T>
	bool tryLock(const T& shared) {
		return getMutexPtr(shared)->try_lock();
	}
};

CV_EXPORTS class Global : public SharedVariables {
public:
	struct Keys {
		enum Enum {
			LOCK_CONTENTION_CNT,
			LOCK_CONTENTION_RATE,
			PLAN_CNT
		};
	};
private:
	ThreadSafeAnyMap<Keys::Enum> map_;

	CV_EXPORTS static Global* instance_;
	std::mutex threadIDMtx_;
	const std::thread::id defaultThreadID_;
	std::thread::id mainThreadID_;
	bool isFirstRun_ = true;
	std::set<string> once_;
	std::mutex nodeLockMtx_;
	std::map<string, std::pair<std::thread::id, cv::Ptr<std::mutex>>> nodeLockMap_;

	CV_EXPORTS cv::Ptr<std::mutex> getNodeLockInternal(const string& name, const bool owned = true) {
		auto it = nodeLockMap_.find(name);
		if(owned) {
			if(it == nodeLockMap_.end()) {
				auto mtxPtr = cv::makePtr<std::mutex>();
				nodeLockMap_[name] = {std::this_thread::get_id(), mtxPtr};
				return mtxPtr;
			} else {
				auto entry = *it;
				if(entry.second.first == std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		} else {
			if(it != nodeLockMap_.end()) {
				auto entry = *it;
				if(entry.second.first != std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		}
		return nullptr;
	}

	CV_EXPORTS bool invalidateNodeLockInternal(const string& name) {
		auto it = nodeLockMap_.find(name);
		if(it != nodeLockMap_.end()) {
			auto& entry = *it;
			entry.second.second = nullptr;
			return true;
		}
		return false;
	}

	Global() {
		create<false, size_t>(Keys::LOCK_CONTENTION_CNT, 0);
		create<false, double>(Keys::LOCK_CONTENTION_RATE, 0.0);
		create<false, size_t>(Keys::PLAN_CNT, 0);
	}
public:
	template <typename V> const V& get(Keys::Enum k) {
		return map_.get<V>(k);
	}

	template <typename V> void set(Keys::Enum k, V v) {
		map_.set(k, v);
	}

	template <bool Tread, typename V> void create(Keys::Enum k, V v, const std::function<void(const V& val)>& cb = std::function<void(const V& val)>()) {
		map_.create<Tread>(k, v, cb);
	}

	template <typename V> V apply(Keys::Enum k, std::function<V(V&)> f) {
		return map_.apply(k, f);
	}

	CV_EXPORTS void setMainID(const std::thread::id& id) {
		std::lock_guard<std::mutex> lock(threadIDMtx_);
		mainThreadID_ = id;
    }

	CV_EXPORTS bool isMain() {
		std::lock_guard<std::mutex> lock(threadIDMtx_);
		return (mainThreadID_ == defaultThreadID_ || mainThreadID_ == std::this_thread::get_id());
	}

	CV_EXPORTS bool isFirstRun() {
		static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
    	bool f = isFirstRun_;
    	isFirstRun_ = false;
		return f;
    }

	CV_EXPORTS static Global& instance() {
		static std::mutex mtx;
		std::lock_guard guard(mtx);
		if(instance_ == nullptr) {
			instance_ = new Global();
		}
		return *instance_;
	}

	CV_EXPORTS cv::Ptr<std::mutex> tryGetNodeLock(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		return getNodeLockInternal(name, false);
	}

	CV_EXPORTS bool lockNode(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		auto lock = getNodeLockInternal(name);
		if(lock) {
			lock->lock();
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS bool tryUnlockNode(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		auto lock = getNodeLockInternal(name);
		if(lock) {
			lock->unlock();
			invalidateNodeLockInternal(name);
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS size_t countNodeLocks() {
		std::lock_guard guard(nodeLockMtx_);
		size_t cnt = 0;
		for(auto entry : nodeLockMap_) {
			if(entry.second.first == std::this_thread::get_id() && entry.second.second) {
				++cnt;
			}
		}
		return cnt;
	}

	CV_EXPORTS bool once(string name) {
	    static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
		string stem = name.substr(0, name.find_last_of("-"));

		auto it = once_.find(stem);
		if(it != once_.end()) {
			std::cerr << "DENIED" << std::endl;
			return false;
		} else {
			once_.insert(stem);
			std::cerr << "GRANTED" << std::endl;
			return true;
		}
	}
};


class RunState {
	CV_EXPORTS static RunState* instance_;
public:
	struct Keys {
		enum Enum {
			FRAME_COUNT,
			RUN_COUNT,
			START_TIME,
			FPS,
			WORKERS_READY,
			WORKERS_STARTED,
			WORKERS_INDEX,
			FRAMEBUFFER_INDEX,
			LOCKING,
			DISPLAY_READY
		};
	};
private:
	ThreadSafeAnyMap<Keys::Enum> map_;
public:
	RunState() {
		create<false, size_t>(Keys::FRAME_COUNT, 0);
		create<false, size_t>(Keys::RUN_COUNT, 0);
		create<false, uint64_t>(Keys::START_TIME, get_epoch_nanos());
		create<false, double>(Keys::FPS, 0);
		create<false, size_t>(Keys::WORKERS_READY, 0);
		create<false, size_t>(Keys::WORKERS_STARTED, 0);
		create<false, size_t>(Keys::WORKERS_INDEX, 0);
		create<false, size_t>(Keys::FRAMEBUFFER_INDEX, 0);
		create<false, bool>(Keys::LOCKING, false);
		create<false, bool>(Keys::DISPLAY_READY, false);
	}

	static RunState& instance() {
		static std::mutex mtx;
		std::lock_guard guard(mtx);
		if(instance_ == nullptr) {
			instance_ = new RunState();
		}
		return *instance_;
	}

	template <typename V> const V& get(Keys::Enum k) {
		return map_.get<V>(k);
	}

	template <typename V> void set(Keys::Enum k, V v) {
		map_.set(k, v);
	}

	template <bool Tread, typename V> void create(Keys::Enum k, V v, const std::function<void(const V& val)>& cb = std::function<void(const V& val)>()) {
		map_.create<Tread>(k, v, cb);
	}

	template <typename V> V apply(Keys::Enum k, std::function<V(V&)> f) {
		return map_.apply(k, f);
	}
};



CV_EXPORTS void copy_cross(const cv::UMat& src, cv::UMat& dst);


/*!
 * Convenience function to color convert from Scalar to Scalar
 * @param src The scalar to color convert
 * @param code The color converions code
 * @return The color converted scalar
 */
CV_EXPORTS cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code);

/*!
 * Convenience function to check for OpenGL errors. Should only be used via the macro #GL_CHECK.
 * @param file The file path of the error.
 * @param line The file line of the error.
 * @param expression The expression that failed.
 */
CV_EXPORTS void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression);
/*!
 * Convenience macro to check for OpenGL errors.
 */
#ifndef NDEBUG
#define GL_CHECK(expr)                            \
    expr;                                        \
    cv::v4d::gl_check_error(__FILE__, __LINE__, #expr);
#else
#define GL_CHECK(expr)                            \
    expr;
#endif
CV_EXPORTS void init_shaders(unsigned int handles[3], const char* vShader, const char* fShader, const char* outputAttributeName);
CV_EXPORTS void init_fragment_shader(unsigned int handles[2], const char* fshader);
/*!
 * Returns the OpenGL vendor string
 * @return a string object with the OpenGL vendor information
 */
CV_EXPORTS std::string getGlVendor();
/*!
 * Returns the OpenGL Version information.
 * @return a string object with the OpenGL version information
 */
CV_EXPORTS std::string getGlInfo();
/*!
 * Returns the OpenCL Version information.
 * @return a string object with the OpenCL version information
 */
CV_EXPORTS std::string getClInfo();
/*!
 * Determines if Intel VAAPI is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isIntelVaSupported();
/*!
 * Determines if cl_khr_gl_sharing is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isClGlSharingSupported();
/*!
 * Tells the application if it's alright to keep on running.
 * Note: If you use this mechanism signal handlers are installed
 * @return true if the program should keep on running
 */
CV_EXPORTS bool keepRunning();

CV_EXPORTS void requestFinish();

CV_EXPORTS void resizePreserveAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize, const cv::Scalar& bgcolor = {0,0,0,255});

}
}

#endif /* SRC_OPENCV_V4D_UTIL_HPP_ */
