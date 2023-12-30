// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include "cubescene.hpp"

using namespace cv::v4d;

class CubeDemoPlan : public Plan {
	CubeScene scene_;
public:
	void setup() override {
		gl(&CubeScene::init, RW(scene_));
	}

	void infer() override {
		gl(&CubeScene::render, R(scene_), V(true));
	}

	void teardown() override {
		gl(&CubeScene::destroy, RW(scene_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 1280, 720);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Cube Demo", AllocateFlags::IMGUI, ConfigFlags::DEFAULT, DebugFlags::LOWER_WORKER_PRIORITY);
	Plan::run<CubeDemoPlan>(0);

	return 0;
}
