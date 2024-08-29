// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include "cubescene.hpp"

using namespace cv::v4d;

class ManyCubesDemoPlan : public Plan {
	constexpr static size_t NUMBER_OF_CONTEXTS_ = 10;

	CubeScene scene_;
	size_t currentGlCtx_ = 0;
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void setup() override {
		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i), &CubeScene::init, RW(scene_));
		}
	}

	void infer() override {
		//Render using multiple OpenGL contexts
		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i),
				&CubeScene::render, R(scene_), V(true), V(double(i) / NUMBER_OF_CONTEXTS_));
		}
	}

	void teardown() override {
		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i), &CubeScene::destroy, R(scene_));
		}
	}
};

int main() {
	cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Many Cubes Demo", AllocateFlags::IMGUI);
    Plan::run<ManyCubesDemoPlan>(0);

    return 0;
}
