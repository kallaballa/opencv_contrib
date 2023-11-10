// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <functional>
#include "../../include/opencv2/v4d/util.hpp"

#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_V4DCONTEXT_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_V4DCONTEXT_HPP_

namespace cv {
namespace v4d {
namespace detail {

class V4DContext {
public:
	virtual ~V4DContext() {}
    virtual void execute(std::function<void()> fn) = 0;
};

class SingleContext : public V4DContext {
public:
	virtual ~SingleContext() {}
    virtual void execute(std::function<void()> fn) override {
    	fn();
    }
};

class OnceContext : public V4DContext {
	std::once_flag flag;
public:
	virtual ~OnceContext() {}
    virtual void execute(std::function<void()> fn) override {
    	std::call_once(flag, fn);
    }
};


class ParallelContext : public V4DContext {
public:
	virtual ~ParallelContext() {}
    virtual void execute(std::function<void()> fn) override {
    	fn();
    }
};

}
}
}

#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_V4DCONTEXT_HPP_ */
