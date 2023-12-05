// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

int v4d_cube_main();
int v4d_many_cubes_main();
int v4d_video_main(int argc, char **argv);
int v4d_nanovg_main(int argc, char **argv);
int v4d_shader_main(int argc, char **argv);
int v4d_font_main();
int v4d_pedestrian_main(int argc, char **argv);
int v4d_optflow_main(int argc, char **argv);
int v4d_beauty_main(int argc, char **argv);
#define main v4d_cube_main
#include "cube-demo.cpp"
#undef main
#define main v4d_many_cubes_main
#include "many_cubes-demo.cpp"
#undef main
#define main v4d_video_main
#include "video-demo.cpp"
#undef main
#define main v4d_nanovg_main
#include "nanovg-demo.cpp"
#undef main
#define main v4d_shader_main
#include "shader-demo.cpp"
#undef main
#define main v4d_font_main
#include "font-demo.cpp"
#undef main
#define main v4d_pedestrian_main
#include "pedestrian-demo.cpp"
#undef main
#define main v4d_optflow_main
#include "optflow-demo.cpp"
#undef main
#define main v4d_beauty_main
#include "beauty-demo.cpp"
#undef main

class MontageDemoPlan : public Plan {
	const cv::Size tiling_  = cv::Size(3, 3);
	const cv::Size tileSz_ = cv::Size(640, 360);
	const cv::Rect tileViewport_ = cv::Rect(0, 720, 640, 360);

	std::vector<Plan*> plans_ = {
		new CubeDemoPlan(tileViewport_),
		new ManyCubesDemoPlan(tileViewport_),
		new VideoDemoPlan(tileViewport_),
		new NanoVGDemoPlan(tileViewport_),
		new ShaderDemoPlan(tileViewport_),
		new FontDemoPlan(tileViewport_),
		new PedestrianDemoPlan(tileViewport_),
		new BeautyDemoPlan(tileViewport_),
		new OptflowDemoPlan(tileViewport_)
	};
	struct Frames {
		std::vector<cv::UMat> results_ = std::vector<cv::UMat>(9);
		cv::UMat captured;
	} frames_;

	cv::Size_<float> scale_;
public:
	MontageDemoPlan(const cv::Rect& vp) : Plan(vp) {
		CV_Assert(plans_.size() == frames_.results_.size() &&  plans_.size() == size_t(tiling_.width * tiling_.height));
	}

	virtual void setup(cv::Ptr<V4D> window) override {
		window->setFramebufferViewport(tileViewport_);
		for(auto* plan : plans_) {
			window->setPrefix(std::to_string((size_t)plan));
			plan->setup(window);
		}
	}

	virtual void infer(cv::Ptr<V4D> window) override {
		window->setFramebufferViewport(tileViewport_);
		window->setPrefix("");
		window->capture();
		window->setDisableIO(true);
		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, cv::UMat& captured){
			cv::resize(framebuffer, captured, tileSize);
		}, tileSz_, frames_.captured);


		for(size_t i = 0; i < plans_.size(); ++i) {
			auto* plan = plans_[i];
			window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
				framebuffer = cv::Scalar::all(0);
				captured.copyTo(framebuffer);
			}, tileSz_, frames_.captured);
			window->setPrefix(std::to_string((size_t)plan));
			plan->infer(window);
			window->setPrefix("");
			window->fb([](const cv::UMat& framebuffer, cv::UMat& result){
				framebuffer.copyTo(result);
			}, frames_.results_[i]);
		}

		cerr << viewport() << endl;
		window->setFramebufferViewport(viewport());
		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSz, const Frames& frames){
			int w = tileSz.width;
			int h = tileSz.height;
			framebuffer = cv::Scalar::all(0);

			for(size_t x = 0; x < 3; ++x)
				for(size_t y = 0; y < 3; ++y)
					frames.results_[x * 3 + y].copyTo(framebuffer(cv::Rect(w * x, h * y, w, h)));
		}, tileSz_, frames_);
		window->setFramebufferViewport(tileViewport_);
		window->setDisableIO(false);
		window->write();
	}

	virtual void teardown(cv::Ptr<V4D> window) override {
		window->setFramebufferViewport(tileViewport_);
		for(auto* plan : plans_) {
			window->setPrefix(std::to_string((size_t)plan));
			plan->teardown(window);
		}
	}
};

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: montage-demo <video-file> <number of extra workers>" << endl;
        exit(1);
    }

	cv::Ptr<MontageDemoPlan> plan = new MontageDemoPlan(cv::Rect(0, 0, 1920, 1080));
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Montage Demo", AllocateFlags::ALL);
    //Creates a source from a file or a device
    auto src = Source::make(window, argv[1]);
    window->setSource(src);
    //Creates a writer sink (which might be hardware accelerated)
    auto sink = Sink::make(window, "montage-demo.mkv", 60, plan->size());
    window->setSink(sink);
    window->run(plan, atoi(argv[2]));

    return 0;
}

