#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageNVG : public Plan {
    using K = V4D::Keys; // Key constants for V4D properties

    // Struct to hold image metadata and NanoVG paint object
    struct Image_t {
        std::string filename_; // Image file name
        nvg::Paint paint_;     // NanoVG paint object for the image
        int w_;                // Image width
        int h_;                // Image height
    } image_;

public:
    // Constructor to initialize the image file name
    DisplayImageNVG(const std::string& filename) {
        image_.filename_ = filename;
    }

    // Setup phase: Create the NanoVG context and load the image
    void setup() override {
        nvg([](Image_t& img) {
            using namespace cv::v4d::nvg;

            // Load the image and get a NanoVG handle
            int handle = createImage(img.filename_.c_str(), NVG_IMAGE_NEAREST);
            CV_Assert(handle > 0); // Ensure the image was loaded successfully

            // Retrieve the image dimensions
            imageSize(handle, &img.w_, &img.h_);

            // Create a NanoVG paint object using the loaded image
            img.paint_ = imagePattern(0, 0, img.w_, img.h_, 0.0f / 180.0f * NVG_PI, handle, 1.0);
        }, RW(image_)); // `RW` denotes read-write access to the shared image data
    }

    // Inference phase: Render the loaded image to the screen
    void infer() override {
        nvg([](const cv::Rect& vp, const Image_t& img) {
            using namespace cv::v4d::nvg;

            beginPath();

            // Scale further rendering calls to match the viewport size
            scale(double(vp.width) / img.w_, double(vp.height) / img.h_);

            // Create a rounded rectangle matching the scaled image dimensions
            roundedRect(0, 0, img.w_, img.h_, 50);

            // Fill the rectangle with the loaded image pattern
            fillPaint(img.paint_);
            fill();
        }, P<cv::Rect>(K::VIEWPORT), RW(image_)); // Pass viewport and image data to the graph node
    }
};

int main() {
    // Define the viewport dimensions
    cv::Rect viewport(0, 0, 960, 960);

    // Initialize the V4D runtime with NanoVG and IMGUI subsystems
    Ptr<V4D> runtime = V4D::init(viewport, "Display an image using NanoVG", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

    // Run the Plan with the specified image file
    Plan::run<DisplayImageNVG>(0, samples::findFile("lena.jpg"));
}
