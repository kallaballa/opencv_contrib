#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageFB : public Plan {
	UMat image_;
	UMat converted_;
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	DisplayImageFB(const string& filename) {
		//Loads an image as a UMat (just in case we have hardware acceleration available)
		imread(filename).copyTo(image_);
	}

	void setup() override {
		plain([](const cv::Rect& vp, cv::UMat& image, cv::UMat& converted) {
			//We have to manually resize and color convert the image when using direct frambuffer access.
			resize(image, converted, vp.size());
			cvtColor(converted, converted, COLOR_RGB2BGRA);
		}, vp_, A(image_), A(converted_));
	}

	void infer() override {
		//Create a fb context and copies the prepared image to the framebuffer. The fb context
		//takes care of retrieving and storing the data on the graphics card (using CL-GL
		//interop if available), ready for other contexts to use
		fb([](UMat& framebuffer, const cv::UMat& c){
			c.copyTo(framebuffer);
		}, A(converted_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960,960);
	//Creates a V4D object
    Ptr<V4D> runtime = V4D::init(viewport, "Display an Image through direct FB access", AllocateFlags::IMGUI);
    Plan::run<DisplayImageFB>(0, samples::findFile("lena.jpg"));

    return 0;
}
