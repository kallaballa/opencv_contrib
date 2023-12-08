// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;


class BgfxDemoPlan : public Plan {
	struct PosColorVertex
	{
		float m_x;
		float m_y;
		float m_z;
		uint32_t m_abgr;

		static void init()
		{
			ms_layout
				.begin()
				.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
				.add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
				.end();
		};

		inline static bgfx::VertexLayout ms_layout;
	};

	inline static PosColorVertex s_cubeVertices[] =
	{
		{-1.0f,  1.0f,  1.0f, 0xff000000 },
		{ 1.0f,  1.0f,  1.0f, 0xff0000ff },
		{-1.0f, -1.0f,  1.0f, 0xff00ff00 },
		{ 1.0f, -1.0f,  1.0f, 0xff00ffff },
		{-1.0f,  1.0f, -1.0f, 0xffff0000 },
		{ 1.0f,  1.0f, -1.0f, 0xffff00ff },
		{-1.0f, -1.0f, -1.0f, 0xffffff00 },
		{ 1.0f, -1.0f, -1.0f, 0xffffffff },
	};

	inline static const uint16_t s_cubeTriList[] =
	{
		0, 1, 2, // 0
		1, 3, 2,
		4, 6, 5, // 2
		5, 6, 7,
		0, 2, 4, // 4
		4, 2, 6,
		1, 5, 3, // 6
		5, 7, 3,
		0, 4, 1, // 8
		4, 5, 1,
		2, 3, 6, // 10
		6, 3, 7,
	};

	inline static const uint16_t s_cubeTriStrip[] =
	{
		0, 1, 2,
		3,
		7,
		1,
		5,
		0,
		4,
		2,
		6,
		7,
		4,
		5,
	};

	inline static const uint16_t s_cubeLineList[] =
	{
		0, 1,
		0, 2,
		0, 4,
		1, 3,
		1, 5,
		2, 3,
		2, 6,
		3, 7,
		4, 5,
		4, 6,
		5, 7,
		6, 7,
	};

	inline static const uint16_t s_cubeLineStrip[] =
	{
		0, 2, 3, 1, 5, 7, 6, 4,
		0, 2, 6, 4, 5, 7, 3, 1,
		0,
	};

	inline static const uint16_t s_cubePoints[] =
	{
		0, 1, 2, 3, 4, 5, 6, 7
	};

	inline static const char* s_ptNames[]
	{
		"Triangle List",
		"Triangle Strip",
		"Lines",
		"Line Strip",
		"Points",
	};

	inline static const uint64_t s_ptState[]
	{
		UINT64_C(0),
		BGFX_STATE_PT_TRISTRIP,
		BGFX_STATE_PT_LINES,
		BGFX_STATE_PT_LINESTRIP,
		BGFX_STATE_PT_POINTS,
	};

	struct Params {
		uint32_t m_width;
		uint32_t m_height;
		bgfx::VertexBufferHandle m_vbh;
		bgfx::IndexBufferHandle m_ibh[BX_COUNTOF(s_ptState)];
		bgfx::ProgramHandle m_program;
		int32_t m_pt = 0;

		bool m_r = true;
		bool m_g = true;
		bool m_b = true;
		bool m_a = true;
	} params_;

	inline static int64_t time_offset_;
public:
	BgfxDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
		Global::registerShared(time_offset_);
	}

	void setup(cv::Ptr<V4D> window) override {
		window->branch(BranchType::ONCE, always_);
		{
			window->plain([](int64_t& timeOffset) {
				timeOffset = bx::getHPCounter();
			}, time_offset_);
		}
		window->endbranch(BranchType::ONCE, always_);
		window->bgfx([](const cv::Rect& vp, Params& params){
			params.m_width = vp.width;
			params.m_height = vp.height;
			// Set view 0 clear state.
			bgfx::setViewClear(0
				, BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
				, 0x00000000
				, 1.0f
				, 0
				);
			PosColorVertex::init();

			// Set view 0 default viewport.
			bgfx::setViewRect(0, vp.x, vp.y, uint16_t(vp.width), uint16_t(vp.height));

			// Create static vertex buffer.
			params.m_vbh = bgfx::createVertexBuffer(
				// Static data can be passed with bgfx::makeRef
				  bgfx::makeRef(s_cubeVertices, sizeof(s_cubeVertices) )
				, PosColorVertex::ms_layout
				);

			// Create static index buffer for triangle list rendering.
			params.m_ibh[0] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(s_cubeTriList, sizeof(s_cubeTriList) )
				);

			// Create static index buffer for triangle strip rendering.
			params.m_ibh[1] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(s_cubeTriStrip, sizeof(s_cubeTriStrip) )
				);

			// Create static index buffer for line list rendering.
			params.m_ibh[2] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(s_cubeLineList, sizeof(s_cubeLineList) )
				);

			// Create static index buffer for line strip rendering.
			params.m_ibh[3] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(s_cubeLineStrip, sizeof(s_cubeLineStrip) )
				);

			// Create static index buffer for point list rendering.
			params.m_ibh[4] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(s_cubePoints, sizeof(s_cubePoints) )
				);

			// Create program from shaders.
			params.m_program = util::load_program("vs_cubes", "fs_cubes");

		}, viewport(), params_);
	}

	void infer(cv::Ptr<V4D> window) override {
		window->capture();
		window->bgfx([](const Params& params, const int64_t& timeOffset){
			float time = (float)( (bx::getHPCounter()-Global::safe_copy(timeOffset))/double(bx::getHPFrequency() ) );

			const bx::Vec3 at  = { 0.0f, 0.0f,   0.0f };
			const bx::Vec3 eye = { 0.0f, 0.0f, -35.0f };

			// Set view and projection matrix for view 0.
			{
				float view[16];
				bx::mtxLookAt(view, eye, at);

				float proj[16];
				bx::mtxProj(proj, 60.0f, float(params.m_width)/float(params.m_height), 0.1f, 100.0f, bgfx::getCaps()->homogeneousDepth);
				bgfx::setViewTransform(0, view, proj);

				// Set view 0 default viewport.
				bgfx::setViewRect(0, 0, 0, uint16_t(params.m_width), uint16_t(params.m_height) );
			}

			// This dummy draw call is here to make sure that view 0 is cleared
			// if no other draw calls are submitted to view 0.
			bgfx::touch(0);

			bgfx::IndexBufferHandle ibh = params.m_ibh[params.m_pt];
			uint64_t state = 0
				| (params.m_r ? BGFX_STATE_WRITE_R : 0)
				| (params.m_g ? BGFX_STATE_WRITE_G : 0)
				| (params.m_b ? BGFX_STATE_WRITE_B : 0)
				| (params.m_a ? BGFX_STATE_WRITE_A : 0)
				| BGFX_STATE_WRITE_Z
				| BGFX_STATE_DEPTH_TEST_LESS
				| BGFX_STATE_CULL_CW
				| BGFX_STATE_MSAA
				| s_ptState[params.m_pt]
				;

			// Submit 11x11 cubes.
			for (uint32_t yy = 0; yy < 11; ++yy)
			{
				for (uint32_t xx = 0; xx < 11; ++xx)
				{
					float mtx[16];
					bx::mtxRotateXY(mtx, time + xx*0.21f, time + yy*0.37f);
					mtx[12] = -15.0f + float(xx)*3.0f;
					mtx[13] = -15.0f + float(yy)*3.0f;
					mtx[14] = 0.0f;

					// Set model matrix for rendering.
					bgfx::setTransform(mtx);

					// Set vertex and index buffer.
					bgfx::setVertexBuffer(0, params.m_vbh);
					bgfx::setIndexBuffer(ibh);

					// Set render states.
					bgfx::setState(state);

					// Submit primitive for rendering to view 0.
					bgfx::submit(0, params.m_program);
				}
			}

			// Advance to next frame. Rendering thread will be kicked to
			// process submitted rendering primitives.
			bgfx::frame();
		}, params_, time_offset_);
		window->write();
	}
};


int main(int argc, char** argv) {
	CV_Assert(argc == 3);
	cv::Ptr<BgfxDemoPlan> plan = new BgfxDemoPlan(cv::Rect(0,0, 1920, 1080));
	cv::Ptr<V4D> window = V4D::make(plan->size(), "Bgfx Demo", AllocateFlags::ALL);
	auto source = Source::make(window, argv[1]);
	auto sink = Sink::make(window, "bgfx-demo.mkv", 60, plan->size());
	window->setSource(source);
	window->setSink(sink);
    window->run(plan, std::stoi(argv[2]));

    return 0;
}
