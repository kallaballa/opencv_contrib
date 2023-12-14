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

class BlankPlan : public Plan {
public:
	using Plan::Plan;
	void infer(cv::Ptr<V4D> window) override {
		window->clear();
	}
};

class MontageDemoPlan : public Plan {
	const bool highresMode_;
	const cv::Size tiling_ = cv::Size(3, 3);
	cv::Size tileSz_;
	cv::Rect tileViewport_;
	std::vector<cv::Rect> targetViewports_;
	std::vector<Plan*> plans_;

	struct Frames {
		std::vector<cv::UMat> results_;
		cv::UMat captured;
	} frames_;

	struct State {
		int32_t zoomed_ = -1;
	};

	static State state_;
public:
	MontageDemoPlan(const cv::Rect& vp, const bool& highresMode) : Plan(vp), highresMode_(highresMode) {
		Global::registerShared(state_);
		tileSz_ = cv::Size(vp.width / tiling_.width, vp.height / tiling_.height);
		if(highresMode_)
			tileViewport_ = vp;
		else
			tileViewport_ = cv::Rect(0, tileSz_.height * (tiling_.height - 1), tileSz_.width, tileSz_.height);

		plans_ = {
				new CubeDemoPlan(tileViewport_),
				new ManyCubesDemoPlan(tileViewport_),
				new VideoDemoPlan(tileViewport_),
				new NanoVGDemoPlan(tileViewport_),
				new ShaderDemoPlan(tileViewport_),
				new FontDemoPlan(tileViewport_),
				new PedestrianDemoPlan(tileViewport_),
				new BlankPlan(tileViewport_),
				new OptflowDemoPlan(tileViewport_)
			};
		CV_Assert(tiling_.width * tiling_.height == plans_.size());
		frames_.results_.resize(plans_.size());
		targetViewports_.resize(plans_.size());

		const int w = tileSz_.width;
		const int h = tileSz_.height;

		for(size_t x = 0; x < 3; ++x) {
			for(size_t y = 0; y < 3; ++y) {
				targetViewports_[x * 3 + y] = cv::Rect(w * x, h * y, w, h);
			}
		}
	}

	void setup(cv::Ptr<V4D> window) override {
		window->setFramebufferViewport(tileViewport_);
		for(auto* plan : plans_) {
			plan->setParentID(id());
			window->setCurrentID(plan->id());
			plan->setup(window);
		}
	}

	void infer(cv::Ptr<V4D> window) override {
		window->setCurrentID(id());
		window->setFramebufferViewport(tileViewport_);
		window->clear();
		window->capture();
		window->setDisableIO(true);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& captured){
			framebuffer.copyTo(captured);
		}, frames_.captured);

		for(size_t i = 0; i < plans_.size(); ++i) {
			auto* plan = plans_[i];
			window->branch(BranchType::PARALLEL, [i](const State& state){
				State s = Global::safe_copy(state);
				return s.zoomed_ == -1 || i == s.zoomed_;
			}, state_);
			{
				window->clear();
				window->fb([](cv::UMat& framebuffer, const cv::UMat& captured){
					captured.copyTo(framebuffer);
				}, frames_.captured);
				window->setCurrentID(plan->id());
				plan->infer(window);
				window->setCurrentID(id());
				window->fb([](const cv::UMat& framebuffer, cv::UMat& result){
					framebuffer.copyTo(result);
				}, frames_.results_[i]);
			}
			window->endBranch();
		}

		window->branch(0, always_)
			->plain([](const cv::Size& sz, const cv::Size& windowSz, const std::vector<cv::Rect>& targetViewports, State& state) {
				{
					using namespace cv::v4d::event;
					const double scaleX = double(sz.width) / windowSz.width;
					const double scaleY = double(sz.height) / windowSz.height;
					const double scale = std::min(scaleX, scaleY);
					Global::Scope scope(state);
					if(state_.zoomed_ > -1) {
						if(consume(Mouse::Type::RELEASE, Mouse::Button::RIGHT)) {
							state_.zoomed_ = -1;
						}
					} else {
						auto events = fetch(Mouse::Type::RELEASE, Mouse::Button::LEFT);
						if(!events.empty()) {
							cv::Point loc = events[0]->position() * scale;
							for(size_t i = 0; i < targetViewports.size(); ++i) {
								if(targetViewports[i].contains(loc)) {
									state.zoomed_ = i;
									break;
								}
							}
						}
					}
				}
			}, size(), window->size(), targetViewports_, state_)
		->endBranch();

		window->setFramebufferViewport(viewport());
		window->clear();
		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSz, const std::vector<cv::Rect>& targetViewports, State& state, const Frames& frames){
			State s = Global::safe_copy(state);

			if(s.zoomed_ > -1) {
				cv::resize(frames.results_[s.zoomed_], framebuffer, framebuffer.size());
			} else {
				for(size_t i = 0; i < targetViewports.size(); ++i) {
					if(!frames.results_[i].empty()) {
						cv::resize(frames.results_[i], framebuffer(targetViewports[i]), tileSz);
					}
				}
			}
		}, tileSz_, targetViewports_, state_, frames_);
		window->setDisableIO(false);
//		window->write();
	}

	void teardown(cv::Ptr<V4D> window) override {
		window->setFramebufferViewport(tileViewport_);
		for(auto* plan : plans_) {
			window->setCurrentID(plan->id());
			plan->teardown(window);
		}
	}
};

MontageDemoPlan::State MontageDemoPlan::state_;

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: montage-demo <video-file> <number of extra workers>" << endl;
        exit(1);
    }
	cv::Ptr<MontageDemoPlan> plan = new MontageDemoPlan(cv::Rect(0, 0, 1280, 720), false);
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Montage Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
    window->setFullscreen(true);
    auto src = Source::make(window, argv[1]);
//    auto sink = Sink::make(window, "montage-demo.mkv", 60, plan->size());
    window->setSource(src);
//    window->setSink(sink);
    window->run(plan, atoi(argv[2]), false);

    return 0;
}

