// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include "cubescene.hpp"

using std::cerr;
using std::endl;

using namespace cv::v4d;

class VideoDemoPlan: public Plan {
private:
	CubeScene scene_;
public:
	void setup() override {
		gl(&CubeScene::init, RW(scene_));
	}

	void infer() override {
		capture();
		gl(&CubeScene::render, R(scene_), V(false));
		write();
	}

	void teardown() override {
		gl(&CubeScene::destroy, RW(scene_));
	}
};


int main(int argc, char** argv) {
	if (argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }

	cv::Rect viewport(0,0,1280,720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Video Demo", AllocateFlags::IMGUI);
    auto src = Source::make(runtime, argv[1]);
    auto sink = Sink::make(runtime, "video-demo.mkv", src->fps(), viewport.size());
    runtime->setSource(src);
    runtime->setSink(sink);
    Plan::run<VideoDemoPlan>(0);

    return 0;
}
