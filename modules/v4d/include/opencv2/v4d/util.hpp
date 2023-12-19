// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_UTIL_HPP_
#define SRC_OPENCV_V4D_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"
#include "threadsafemap.hpp"
#include "detail/framebuffercontext.hpp"
#include <filesystem>
#include <string>
#include <iostream>
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

inline uint64_t get_epoch_nanos() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

typedef std::latch Latch;
typedef cv::Ptr<std::latch> LatchPtr;

class CV_EXPORTS Global {
	friend class cv::v4d::V4D;
	friend class cv::v4d::detail::FrameBufferContext;
public:
	enum Keys {
		FRAME_COUNT,
		RUN_COUNT,
		START_TIME,
		FPS,
		WORKERS_READY,
		WORKERS_STARTED,
		WORKERS_INDEX,
		FRAMEBUFFER_INDEX,
		LOCKING,
		WORKER_READY_BARRIER,
		DISPLAY_READY
	};

	Global() {
		reset();
	}

	template <typename T>
	class Scope {
	private:
		const T& t_;
	public:

		Scope(const T& t) : t_(t) {
			lock(t_);
		}

		~Scope() {
			unlock(t_);
		}
	};

	template<typename T>
	static bool isShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		return shared_.find(reinterpret_cast<size_t>(&shared)) != shared_.end();
	}

	template<typename T>
	static void registerShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		shared_.insert(std::make_pair(reinterpret_cast<size_t>(&shared), new std::mutex()));
	}

	template<typename T>
	static void safe_copy(const T& from, T& to) {
		std::mutex* mtx = nullptr;
		{
			std::lock_guard<std::mutex> guard(sharedMtx_);
			auto itFrom = shared_.find(reinterpret_cast<size_t>(&from));
			auto itTo = shared_.find(reinterpret_cast<size_t>(&to));

			if(itFrom != shared_.end()) {
				mtx = (*itFrom).second;
			} else if(itTo != shared_.end()) {
				mtx = (*itTo).second;
			} else {
				throw std::runtime_error("You are unnecessarily safe-copying a variable or you forgot to register it.");
			}
		}
		{
			CV_Assert(mtx);
			std::lock_guard<std::mutex> guard(*mtx);
			to = copy(from);
		}
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

	static double fps() {
		return get<double>(FPS);
	}

	static size_t workers_started() {
		return get<size_t>(WORKERS_STARTED);
	}
private:
	static ThreadSafeMap<Keys> map_;
	static std::mutex global_mtx_;
	static bool is_first_run_;
	static std::set<string> once_;

	static std::mutex thread_id_mtx_;
	static const std::thread::id default_thread_id_;
	static std::thread::id main_thread_id_;
	static thread_local bool is_main_;

	static std::mutex sharedMtx_;
	static std::map<size_t, std::mutex*> shared_;
	static std::mutex node_lock_mtx_;
	static std::map<string, std::pair<std::thread::id, cv::Ptr<std::mutex>>> node_lock_map_;
	typedef typename std::map<size_t, std::mutex*>::iterator ThreadWorkerIdIterator;

	template<typename T>
	static T copy_construct(const T& t) {
		return t;
	}

	CV_EXPORTS static void reset() {
		set<size_t>(FRAME_COUNT, 0);
		set<size_t>(RUN_COUNT, 0);
		set<uint64_t>(START_TIME, get_epoch_nanos());
		set<double>(FPS, 0);
		set<size_t>(WORKERS_READY, 0);
		set<size_t>(WORKERS_STARTED, 0);
		set<size_t>(WORKERS_INDEX, 0);
		set<size_t>(FRAMEBUFFER_INDEX, 0);
		set<bool>(LOCKING, false);
		set<bool>(DISPLAY_READY, false);
	}

	CV_EXPORTS static cv::Ptr<std::mutex> get_node_lock_internal(const string& name, const bool owned = true) {
		auto it = node_lock_map_.find(name);
		if(owned) {
			if(it == node_lock_map_.end()) {
				auto mtxPtr = cv::makePtr<std::mutex>();
				node_lock_map_[name] = {std::this_thread::get_id(), mtxPtr};
				return mtxPtr;
			} else {
				auto entry = *it;
				if(entry.second.first == std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		} else {
			if(it != node_lock_map_.end()) {
				auto entry = *it;
				if(entry.second.first != std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		}
		return nullptr;
	}

	CV_EXPORTS static bool invalidate_node_lock_internal(const string& name) {
		auto it = node_lock_map_.find(name);
		if(it != node_lock_map_.end()) {
			auto& entry = *it;
			entry.second.second = nullptr;
			return true;
		}
		return false;
	}
	CV_EXPORTS static cv::Ptr<std::mutex> try_get_node_lock(const string& name) {
		std::lock_guard guard(node_lock_mtx_);
		return get_node_lock_internal(name, false);
	}

	CV_EXPORTS static bool lock_node(const string& name) {
		std::lock_guard guard(node_lock_mtx_);
		auto lock = get_node_lock_internal(name);
		if(lock) {
			lock->lock();
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS static bool try_unlock_node(const string& name) {
		std::lock_guard guard(node_lock_mtx_);
		auto lock = get_node_lock_internal(name);
		if(lock) {
			lock->unlock();
			invalidate_node_lock_internal(name);
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS static size_t cound_node_locks() {
		std::lock_guard guard(node_lock_mtx_);
		size_t cnt = 0;
		for(auto entry : node_lock_map_) {
			if(entry.second.first == std::this_thread::get_id() && entry.second.second) {
				++cnt;
			}
		}
		return cnt;
	}

	CV_EXPORTS static std::mutex& mutex() {
    	return global_mtx_;
    }

	CV_EXPORTS static bool once(string name) {
	    static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
		string stem = name.substr(0, name.find_last_of("-"));
		if(once_.empty()) {
			once_.insert(stem);
			return true;
		}

		auto it = once_.find(stem);
		if(it != once_.end()) {
			return false;
		} else {
			once_.insert(stem);
			return true;
		}
	}

	CV_EXPORTS static void set_main_id(const std::thread::id& id) {
		std::lock_guard<std::mutex> lock(thread_id_mtx_);
		main_thread_id_ = id;
    }

	CV_EXPORTS static bool is_main() {
		std::lock_guard<std::mutex> lock(thread_id_mtx_);
		return (main_thread_id_ == default_thread_id_ || main_thread_id_ == std::this_thread::get_id());
	}

	CV_EXPORTS static bool is_first_run() {
		static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
    	bool f = is_first_run_;
    	is_first_run_ = false;
		return f;
    }

	template<typename T>
	static void lock(const T& shared) {
		ThreadWorkerIdIterator it, end;
		std::mutex* mtx = nullptr;
		{
			std::lock_guard<std::mutex> guard(sharedMtx_);
			it = shared_.find(reinterpret_cast<size_t>(&shared));
			end = shared_.end();
			if(it != end) {
				mtx = (*it).second;
			}
		}

		if(mtx != nullptr) {
			mtx->lock();
			return;
		}
		throw std::runtime_error("You are trying to lock a non-shared variable");
	}

	template<typename T>
	static void unlock(const T& shared) {
		ThreadWorkerIdIterator it, end;
		std::mutex* mtx = nullptr;
		{
			std::lock_guard<std::mutex> guard(sharedMtx_);
			it = shared_.find(reinterpret_cast<size_t>(&shared));
			end = shared_.end();
			if(it != end) {
				mtx = (*it).second;
			}
		}

		if(mtx != nullptr) {
			mtx->unlock();
			return;
		}

		throw std::runtime_error("You are trying to unlock a non-shared variable");
	}

	template <typename V> static V get(Keys k) {
		return map_.get<V>(k);
	}

	template <typename V> static void set(Keys k, V v) {
		map_.set(k, v);
	}

	template <typename V> static V on(Keys k, std::function<V(V&)> f) {
		return map_.on(k, f);
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
