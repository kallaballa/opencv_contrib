#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

// A Plan implementation that renders a blue screen using OpenGL
class RenderOpenGLPlan : public Plan {
public:
    // Setup phase of inference: Creates graph nodes that run once at the start of the algorithm's lifetime
    void setup() override {
        // Sets the clear color to blue by creating a graph node with an OpenGL context (provided by V4D)
        // "gl" is a context-call that provides resources to the graph node
        // These resources may be shared, requiring locking
        // V4D can create multiple OpenGL contexts in parallel via an overload of "gl"
        // "V" is an edge-call that provides constants to the algorithm
        // Other edge-calls provide read access (R), read-write access (RW), and access by copy (C)
        // There are variants of these edge-calls for shared data (RS, RWS, CS)
        // Fine-grained definition of edge-calls (using R over RW where possible,
        // breaking down code into shared and non-shared sections) helps Plan build an optimal graph
        // Edge-calls have special support for smart pointers and cv::UMat objects
        gl(glClearColor, V(0), V(0), V(1), V(1));
    }

    // Main phase of inference: Creates graph nodes that run in a loop after the nodes created by the setup phase have run
    void infer() override {
        // Clears the screen. The clear color and other GL states are preserved between context-calls
        gl(glClear, V(GL_COLOR_BUFFER_BIT));
    }
};

int main() {
    // The viewport may be changed at runtime by creating a set node (via a "set" call)
    cv::Rect viewport(0, 0, 960, 960);
    // Initialization of the V4D runtime must be invoked before Plan::run is called
    // There are AllocateFlags for selective initialization of subsystems, ConfigFlags, and DebugFlags
    Ptr<V4D> runtime = V4D::init(viewport, "GL Blue Screen", AllocateFlags::IMGUI);
    // Build (infer) and run the graph. The number denotes the number of workers (0 meaning auto, which currently resolves to 1)
    Plan::run<RenderOpenGLPlan>(0);
}
