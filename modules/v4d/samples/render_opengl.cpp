#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class RenderOpenGLPlan : public Plan {
public:
	void setup() override {
		//Sets the clear color to blue
		gl(glClearColor, V(0), V(0), V(1), V(1));
	}
	void infer() override {
		//Clears the screen. The clear color and other GL-states are preserved between context-calls.
		gl(glClear, V(GL_COLOR_BUFFER_BIT));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> runtime = V4D::init(viewport, "GL Blue Screen", AllocateFlags::IMGUI,  ConfigFlags::DEFAULT, DebugFlags::DEFAULT);
    Plan::run<RenderOpenGLPlan>(0);
}

