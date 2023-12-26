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
	void infer() override {
		clear();
	}
};

class MontageDemoPlan : public Plan {
	using K = V4D::Keys;
	const bool highresMode_;
	const cv::Size tiling_ = cv::Size(3, 3);
	cv::Size tileSz_;
	cv::Rect tileViewport_;
	std::vector<cv::Rect> targetViewports_;
	std::vector<cv::Ptr<Plan>> plans_;
	std::vector<cv::Ptr<Plan>> highResPlans_;

	Property<cv::Size> winSz_ = GET<cv::Size>(K::WINDOW_SIZE);
	Property<cv::Rect> initVp_ = GET<cv::Rect>(K::INIT_VIEWPORT);

	struct Frames {
		std::vector<cv::UMat> results_;
		cv::UMat captured;
	} frames_;

	struct State {
		int32_t zoomed_ = -1;
	};

	static State state_;
	string id_;
public:
	MontageDemoPlan(const cv::Rect& vp, const bool& highresMode) : highresMode_(highresMode) {
		_shared(state_);

		tileSz_ = cv::Size(vp.width / tiling_.width, vp.height / tiling_.height);
		if(highresMode_)
			tileViewport_ = vp;
		else
			tileViewport_ = cv::Rect(0, 0, tileSz_.width, tileSz_.height);

		plans_ = {
				Plan::makeSubPlan<CubeDemoPlan>(*this),
				Plan::makeSubPlan<ManyCubesDemoPlan>(*this),
				Plan::makeSubPlan<VideoDemoPlan>(*this),
				Plan::makeSubPlan<NanoVGDemoPlan>(*this),
				Plan::makeSubPlan<ShaderDemoPlan>(*this),
				Plan::makeSubPlan<FontDemoPlan>(*this),
				Plan::makeSubPlan<PedestrianDemoPlan>(*this),
				Plan::makeSubPlan<BeautyDemoPlan>(*this),
				Plan::makeSubPlan<OptflowDemoPlan>(*this)
			};


		highResPlans_= {
				Plan::makeSubPlan<CubeDemoPlan>(*this),
				Plan::makeSubPlan<ManyCubesDemoPlan>(*this),
				Plan::makeSubPlan<VideoDemoPlan>(*this),
				Plan::makeSubPlan<NanoVGDemoPlan>(*this),
				Plan::makeSubPlan<ShaderDemoPlan>(*this),
				Plan::makeSubPlan<FontDemoPlan>(*this),
				Plan::makeSubPlan<PedestrianDemoPlan>(*this),
				Plan::makeSubPlan<BeautyDemoPlan>(*this),
				Plan::makeSubPlan<OptflowDemoPlan>(*this)
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

	void setup() override {
		set(K::VIEWPORT, R(tileViewport_));
		for(auto plan : plans_) {
			subSetup(plan);
		}

		set(K::VIEWPORT, initVp_);
		for(auto hrPlan : highResPlans_) {
			subSetup(hrPlan);
		}
	}

	void infer() override {
		set(K::VIEWPORT, initVp_);

		clear()
		->capture();

		set(K::DISABLE_VIDEO_IO, VAL(true));

		fb([](const cv::UMat& framebuffer, cv::UMat& captured){
			framebuffer.copyTo(captured);
		}, RW(frames_.captured));

		branch(BranchType::PARALLEL, [](const State& state){
			return state.zoomed_ == -1;
		}, R_SC(state_));
		{
			for(size_t i = 0; i < plans_.size(); ++i) {
				auto plan = plans_[i];
				set(K::VIEWPORT,VAL(tileViewport_))
				->fb([](cv::UMat& framebuffer, const cv::Rect tileViewPort, const cv::UMat& captured){
					cv::resize(captured, framebuffer, tileViewPort.size());
				}, R(tileViewport_), R(frames_.captured))
				->subInfer(plan)
				->fb([](const cv::UMat& framebuffer, const size_t& idx, Frames& frames) {
					framebuffer.copyTo(frames.results_[idx]);
				}, VAL(i), RW(frames_));
			}
		}
		elseBranch();
		{
			for(size_t i = 0; i < highResPlans_.size(); ++i) {
				auto hrPlan = highResPlans_[i];
				set(K::VIEWPORT, initVp_)
				->branch([](const size_t idx, const State& state){
					return idx == state.zoomed_;
				}, VAL(i), R_SC(state_))
					->fb([](cv::UMat& framebuffer, const cv::UMat& captured){
						captured.copyTo(framebuffer);
					}, R(frames_.captured))
					->subInfer(hrPlan)
					->fb([](const cv::UMat& framebuffer, const size_t& idx, Frames& frames) {
						framebuffer.copyTo(frames.results_[idx]);
					}, VAL(i), RW(frames_))
				->endBranch();
			}
		}
		endBranch();


		branch(BranchType::SINGLE, always_)
			->plain([](const cv::Rect& vp, const cv::Size& windowSz, const std::vector<cv::Rect>& targetViewports, State& state) {
				{
					using namespace cv::v4d::event;
					const double scaleX = double(vp.width) / windowSz.width;
					const double scaleY = double(vp.height) / windowSz.height;
					const double scale = std::min(scaleX, scaleY);
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
			}, initVp_, winSz_, R(targetViewports_), RW_S(state_))
		->endBranch();

		set(K::VIEWPORT, initVp_);
		fb([](cv::UMat& framebuffer, const cv::Size& tileSz, const std::vector<cv::Rect>& targetViewports, const State& state, const Frames& frames) {
			if(state.zoomed_ > -1) {
				cv::resize(frames.results_[state.zoomed_], framebuffer, framebuffer.size());
			} else {
				for(size_t i = 0; i < targetViewports.size(); ++i) {
					if(!frames.results_[i].empty()) {
						cv::resize(frames.results_[i], framebuffer(targetViewports[i]), tileSz);
					}
				}
			}
		}, R(tileSz_), R(targetViewports_), R_SC(state_), R(frames_));

		set(K::DISABLE_VIDEO_IO, VAL(false));
//		window->write();
	}


	void teardown() override {
		set(K::VIEWPORT, R(tileViewport_));
		for(auto plan : plans_) {
			subTeardown(plan);
		}
		set(K::VIEWPORT, initVp_);
		for(auto plan : highResPlans_) {
			subSetup(plan);
		}
	}
};

MontageDemoPlan::State MontageDemoPlan::state_;

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: montage-demo <video-file> <number of extra workers>" << endl;
        exit(1);
    }
	cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Montage Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI, ConfigFlags::DEFAULT, DebugFlags::MONITOR_RUNTIME_PROPERTIES);
    auto src = Source::make(runtime, argv[1]);
    runtime->setSource(src);
    Plan::run<MontageDemoPlan>(atoi(argv[2]), viewport, false);

    return 0;
}

