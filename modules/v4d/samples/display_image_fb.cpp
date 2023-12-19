#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageFB : public Plan {
	UMat image_;
	UMat converted_;
public:
	using Plan::Plan;

	void setup(cv::Ptr<V4D> win) override {
		win->plain([](const cv::Size& sz, cv::UMat& image, cv::UMat& converted) {
			//Loads an image as a UMat (just in case we have hardware acceleration available)
			imread(samples::findFile("lena.jpg")).copyTo(image);

			//We have to manually resize and color convert the image when using direct frambuffer access.
			resize(image, converted, sz);
			cvtColor(converted, converted, COLOR_RGB2BGRA);
		}, R(size()), RW(image_), RW(converted_));
	}

	void infer(Ptr<V4D> win) override {
		//Create a fb context and copies the prepared image to the framebuffer. The fb context
		//takes care of retrieving and storing the data on the graphics card (using CL-GL
		//interop if available), ready for other contexts to use
		win->fb([](UMat& framebuffer, const cv::UMat& c){
			c.copyTo(framebuffer);
		}, R(converted_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960,960);
	//Creates a V4D object
    Ptr<V4D> window = V4D::make(viewport.size(), "Display an Image through direct FB access", AllocateFlags::IMGUI);
    window->run<DisplayImageFB>(0, viewport);
}
