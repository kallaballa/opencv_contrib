#include "opencv2/core/cvdef.h"
#define EVENT_API_EXPORT CV_EXPORTS
#include "../include/opencv2/v4d/events.hpp"

namespace cv {
namespace v4d {
namespace event {
namespace detail {
	GLFWwindow* Holder::main_window = nullptr;
;
}
}
}
}
