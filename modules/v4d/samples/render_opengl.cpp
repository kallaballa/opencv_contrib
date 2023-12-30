#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class RenderOpenGLPlan : public Plan {
public:
	void setup() override {
		gl([]() {
			//Sets the clear color to blue
			glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
		});
	}
	void infer() override {
		gl([]() {
			//Clears the screen. The clear color and other GL-states are preserved between context-calls.
			glClear(GL_COLOR_BUFFER_BIT);
		});
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> runtime = V4D::init(viewport, "GL Blue Screen", AllocateFlags::IMGUI,  ConfigFlags::DEFAULT, DebugFlags::LOWER_WORKER_PRIORITY);
    Plan::run<RenderOpenGLPlan>(0);
}

