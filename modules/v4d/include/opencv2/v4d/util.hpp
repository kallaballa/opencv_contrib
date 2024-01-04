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
#include <array>

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

template<auto V1, decltype(V1) V2, typename T>
struct values_equal : std::bool_constant<V1 == V2>
{
    using type = T;
};

template<typename T>
struct default_type : std::true_type
{
    using type = T;
};

//https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
CV_EXPORTS std::string demangle(const char* name);

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
struct is_callable : public std::disjunction<std::disjunction<
									has_call_operator_t<T>,
									std::is_pointer<T>>,
									std::is_member_function_pointer<T>,
									std::is_function<T>> {
};

//https://stackoverflow.com/a/27885283/1884837



template<class T>
struct function_traits : function_traits<decltype(&T::operator())> {
};


template<>
struct function_traits<std::false_type> : std::false_type {
	using result_type = std::false_type;
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
	std::mutex safeVarsMtx_;
	std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>> sharedVars_;
	std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>> safeVars_;
	typedef typename std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>>::iterator SharedVarsIter;

	template<typename T>
	std::pair<size_t, size_t> findSharedParent(const T& shared) {
		off_t varOffset = reinterpret_cast<size_t>(&shared);
		off_t registeredOffset = 0;
		off_t registeredSize = 0;
		for(const auto& p : sharedVars_) {
			registeredOffset = p.first;
			if(varOffset > registeredOffset) {
				registeredSize = p.second.first;
				if(varOffset < (registeredOffset + registeredSize)) {
					return {registeredOffset, registeredSize};
				}
			}
		}
		return {0,0};
	}

public:
	template<typename Tplan, typename Tvar>
	static bool isPlanMember(Tplan& plan, Tvar& var) {
		const char* planPtr = reinterpret_cast<const char*>(&plan);
		const char* varPtr = reinterpret_cast<const char*>(&var);
		off_t parentOffset = plan.getParentOffset();
		off_t parentActualSize = plan.getParentActualTypeSize();
		off_t actualTypeSize = plan.getActualTypeSize();
		off_t varOffset = off_t (varPtr);
		off_t planOffset = off_t (planPtr);
		off_t planSize = sizeof(Tplan);

		CV_Assert((parentOffset == 0  && parentActualSize == 0) || (parentOffset > 0 && parentActualSize > 0));

		off_t parentLowerBound = parentOffset;
		off_t parentUpperBound = parentOffset + parentActualSize;
		off_t lowerBound = planOffset;
		off_t upperBound = planOffset + actualTypeSize;

		if(! ((varOffset >= lowerBound && varOffset <= upperBound) || (parentOffset > 0 && varOffset >= parentLowerBound && varOffset <= parentUpperBound))) {
			return false;
		}
		return true;
	}


	template<typename T>
	void makeSharedVar(const T& candidate) {
		{
			std::lock_guard<std::mutex> guard(safeVarsMtx_);
			CV_Assert(safeVars_.find(reinterpret_cast<size_t>(&candidate)) == safeVars_.end());
		}

		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		if(sharedVars_.find(reinterpret_cast<size_t>(&candidate)) != sharedVars_.end()) {
			return;
		} else  {
			auto parent = findSharedParent(candidate);
			if(parent.first != 0) {
				auto it = sharedVars_.find(parent.first);
				CV_Assert(it != sharedVars_.end());
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), (*it).second.second)});
			} else {
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
			}
		}
	}

	template<typename Tplan, typename T, bool Tcheck = true>
	bool checkShared(Tplan& plan, const T& candidate) {
		{
			std::lock_guard<std::mutex> guard(safeVarsMtx_);
			if(safeVars_.find(reinterpret_cast<size_t>(&candidate)) != safeVars_.end()) {
				return false;
			}
		}

		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		if(sharedVars_.find(reinterpret_cast<size_t>(&candidate)) != sharedVars_.end()) {
			return true;
		} else if(!isPlanMember(plan, candidate)) {
			auto parent = findSharedParent(candidate);
			if(parent.first != 0) {
				auto it = sharedVars_.find(parent.first);
				CV_Assert(it != sharedVars_.end());
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), (*it).second.second)});
			} else {
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
			}

			return true;
		}
		return false;
	}

	template<typename T>
	void registerSafe(const T& safe) {
		std::lock_guard<std::mutex> guard(safeVarsMtx_);
		auto it = safeVars_.find(reinterpret_cast<size_t>(&safe));
		if(it == safeVars_.end()) {
			safeVars_.insert({reinterpret_cast<size_t>(&safe), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
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
			return false;
		} else {
			once_.insert(stem);
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


template<typename T>
constexpr int matrix_depth() {
	if constexpr(std::is_same_v<T, uchar>) {
		return CV_8U;
	} else if constexpr(std::is_same_v<T, short>){
		return CV_16S;
	} else if constexpr(std::is_same_v<T, ushort>){
		return CV_16U;
	} else if constexpr(std::is_same_v<T, int>){
		return CV_32S;
	} else if constexpr(std::is_same_v<T, float>){
		return CV_32F;
	} else if constexpr(std::is_same_v<T, double>){
		return CV_64F;
	} else if constexpr(true) {
		static_assert(false, "Type not supported for operation.");
		return 0;
	}
}

template<bool Tround, typename T> T doRound(T& t) {
	if constexpr(Tround) {
		return std::round(t);
	} else {
		return t;
	}

}
/*!
 * Convenience function to color convert from all Vec_ and Scalar_ variants
 * @param src The color to convert
 * @param code The color converions code
 * @return The color converted scalar
 */
template<int Tcode = -1, typename Tsrc, typename Tdst = Vec<typename Tsrc::value_type, Tsrc::channels>, bool Tround = std::is_floating_point_v<typename Tsrc::value_type> && std::is_integral_v<typename Tdst::value_type>>
Tdst convert_pix(const Tsrc& src, double alpha = 1, double beta = 0) {
	constexpr int srcCn = Tsrc::channels;
	constexpr int dstCn = Tdst::channels;

	using srcv_t = typename Tsrc::value_type;
	using dstv_t = typename Tdst::value_type;
	using src_internal_t = Vec<srcv_t, srcCn>;
	using intermediate_t = Vec<srcv_t, dstCn>;
	using dst_internal_t = Vec<dstv_t, dstCn>;
	static_assert((srcCn == 3 || srcCn == 4) && (dstCn == 3 || dstCn == 4), "Only 3 or 4 (src/dst) channels supported");
	constexpr int srcType = CV_MAKETYPE(matrix_depth<typename src_internal_t::value_type>(), src_internal_t::channels);
	constexpr int intermediateType = CV_MAKETYPE(matrix_depth<typename src_internal_t::value_type>(), dst_internal_t::channels);
	constexpr int dstType = CV_MAKETYPE(matrix_depth<typename dst_internal_t::value_type>(), dst_internal_t::channels);

	std::array<src_internal_t, 1> srcArr;
	if constexpr(srcCn == 3) {
		srcArr[0] = src_internal_t(src[0], src[1], src[2]);
	} else {
		srcArr[0] = src_internal_t(src[0], src[1], src[2], src[3]);
	}

	cv::Mat intermediateMat(cv::Size(1, 1), intermediateType);

	if constexpr(dstCn == srcCn) {
		intermediateMat = srcArr[0];
	} else if (dstCn == 3) {
		intermediateMat =  intermediate_t(srcArr[0][0], srcArr[0][1], srcArr[0][2]);
	} else if (dstCn == 4) {
		intermediateMat = intermediate_t(srcArr[0][0], srcArr[0][1], srcArr[0][2], srcArr[0][3]);
	}

	if constexpr(Tcode >= 0) {
		cvtColor(srcArr, intermediateMat, Tcode);
	}


	std::array<dst_internal_t, 1> dstArr;
	if constexpr(!std::is_same<srcv_t, dstv_t>::value) {
		//will just copy if types match
		intermediateMat.convertTo(dstArr, dstType);
	} else {
		//both value type and channes match
		dstArr[0] = intermediateMat.at<dst_internal_t>(0.0);
	}

	Tdst dst;
	cv::Scalar temp;
	bool doScale = alpha != 1 || beta != 0;
	if(doScale) {
		if constexpr (dstCn == 3) {
			temp = cv::Scalar(dstArr[0][0], dstArr[0][1], dstArr[0][2]);
		} else {
			temp = cv::Scalar(dstArr[0][0], dstArr[0][1], dstArr[0][2], dstArr[0][3]);
		}

		((temp *= alpha) += Scalar::all(beta));

		if constexpr (dstCn == 3) {
			dst = Tdst(doRound<Tround>(temp[0]), doRound<Tround>(temp[1]), doRound<Tround>(temp[2]));
		} else {
			dst = Tdst(doRound<Tround>(temp[0]), doRound<Tround>(temp[1]), doRound<Tround>(temp[2]), doRound<Tround>(temp[3]));
		}
	} else {
		if constexpr (dstCn == 3) {
			dst =  Tdst(doRound<Tround>(dstArr[0][0]), doRound<Tround>(dstArr[0][1]), doRound<Tround>(dstArr[0][2]));
		} else if (dstCn == 4) {
			dst =  Tdst(doRound<Tround>(dstArr[0][0]), doRound<Tround>(dstArr[0][1]), doRound<Tround>(dstArr[0][2]), doRound<Tround>(dstArr[0][3]));
		}
	}

	return dst;
}

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
