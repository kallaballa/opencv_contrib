// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/ocl.hpp>

#include "../include/opencv2/v4d/v4d.hpp"
#include "../include/opencv2/v4d/util.hpp"
#include "../include/opencv2/v4d/detail/gl.hpp"

#include <csignal>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>
#include <regex>

using std::cerr;
using std::endl;

namespace cv {
namespace v4d {
namespace detail {
void print_nth_line(const std::string& str, int n) {
    std::istringstream stream(str);
    std::string line;
    int currentLine = 1;

    while (std::getline(stream, line)) {
        if (currentLine == n) {
            std::cout << line << std::endl;
            return;
        }
        ++currentLine;
    }

    std::cout << "Line " << n << " does not exist." << std::endl;
}
}

size_t cnz(const cv::UMat& m) {
    cv::UMat grey;
    if(m.channels() == 1) {
        grey = m;
    } else if(m.channels() == 3) {
        cvtColor(m, grey, cv::COLOR_BGR2GRAY);
    } else if(m.channels() == 4) {
        cvtColor(m, grey, cv::COLOR_BGRA2GRAY);
    } else {
        assert(false);
    }
    return cv::countNonZero(grey);
}

CV_EXPORTS Global* Global::instance_ = nullptr;
CV_EXPORTS thread_local RunState* RunState::instance_ = nullptr;

CV_EXPORTS void copy_cross(const cv::UMat& src, cv::UMat& dst) {
	if(dst.empty())
		dst.create(src.size(), src.type());
	Mat m = dst.getMat(cv::ACCESS_WRITE);
	src.copyTo(m);
}

void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
    int errorCode = glGetError();
//    cerr << "TRACE: " << file.filename() << " (" << line << ") : " << expression << " => code: " << errorCode << endl;
    if (errorCode != 0) {
        std::stringstream ss;
        ss << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   "
                << expression << "\nError code:\n   " << errorCode;
        CV_LOG_WARNING(nullptr, ss.str());
    }
}

void init_fragment_shader(unsigned int handles[2], const char* fshader) {
    struct Shader {
        GLenum type;
        const char* source;
    } s = { GL_FRAGMENT_SHADER, fshader };

    handles[0] = glCreateProgram();

	;
	handles[1] = glCreateShader(s.type);
	glShaderSource(handles[1] , 1, (const GLchar**) &s.source, NULL);
	glCompileShader(handles[1] );

	GLint compiled;
	glGetShaderiv(handles[1] , GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		std::cerr << " failed to compile:" << std::endl;
		GLint logSize;
		glGetShaderiv(handles[1], GL_INFO_LOG_LENGTH, &logSize);
		char* logMsg = new char[logSize];
		glGetShaderInfoLog(handles[1] , logSize, NULL, logMsg);
        std::cerr <<  logMsg << std::endl;

		delete[] logMsg;

		exit (EXIT_FAILURE);
	}
	glAttachShader(handles[0], handles[1]);

    /* link  and error check */
    glLinkProgram(handles[0]);

    GLint linked;
    glGetProgramiv(handles[0], GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Shader program failed to link: " << fshader << std::endl;
        GLint logSize;
        glGetProgramiv(handles[0], GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog(handles[0], logSize, NULL, logMsg);
        std::cerr <<  logMsg << std::endl;

        delete[] logMsg;

        exit (EXIT_FAILURE);
    }
}

void init_shaders(unsigned int handles[3], const string vShader, const string fShader, const string outputAttributeName) {
#if !defined(OPENCV_V4D_USE_ES3)
	const string version="#version 330\n";
#else
	const string version="#version 300 es\n";
#endif
	string vs = (version + vShader);
	string fs = (version + fShader);
	struct Shader {
        GLenum type;
        const char* source;
    } shaders[2] = { { GL_VERTEX_SHADER, vs.c_str() }, { GL_FRAGMENT_SHADER, fs.c_str() } };

    GLuint program = glCreateProgram();
    handles[0] = program;

    for (int i = 0; i < 2; ++i) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader(s.type);
        handles[i + 1] = shader;
        glShaderSource(shader, 1, (const GLchar**) &s.source, NULL);
        glCompileShader(shader);

        GLint compiled;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            std::cerr << " failed to compile:" << std::endl;
            GLint logSize;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
            char* logMsg = new char[logSize];
            glGetShaderInfoLog(shader, logSize, NULL, logMsg);

            std::cerr << shaders[i].source << std::endl <<  logMsg << std::endl;
            std::regex rex(R"(\d+:(\d+)\(\d+\))");

            std::cmatch cm;
            if (std::regex_search(logMsg, cm, rex)) {
            	for(size_t i = 0; i < cm.size(); ++i)
            		print_nth_line(shaders[i].source, atoi(std::string(cm[1]).c_str()));
            }
            std::cerr << std::endl;

            delete[] logMsg;

            exit (EXIT_FAILURE);
        }

        glAttachShader(program, shader);
    }
#if !defined(OPENCV_V4D_USE_ES3)
    /* Link output */
    glBindFragDataLocation(program, 0, outputAttributeName.c_str());
#else
    CV_UNUSED(outputAttributeName);
#endif
    /* link  and error check */
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
    	std::cerr << "Shader program failed to link: " << fShader << std::endl;
        GLint logSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog(program, logSize, NULL, logMsg);

        std::cerr <<  logMsg << std::endl;

        delete[] logMsg;

        exit (EXIT_FAILURE);
    }

}

std::string get_gl_vendor()  {
    std::ostringstream oss;
        oss << reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        return oss.str();
}

std::string get_gl_info() {
    std::ostringstream oss;
    oss << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_VERSION))
            << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;
    return oss.str();
}

std::string get_cl_info() {
    std::stringstream ss;
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL()) {
		std::vector<cv::ocl::PlatformInfo> plt_info;
		cv::ocl::getPlatfomsInfo(plt_info);
		const cv::ocl::Device& defaultDevice = cv::ocl::Device::getDefault();
		cv::ocl::Device current;
		ss << endl;
		for (const auto& info : plt_info) {
			for (int i = 0; i < info.deviceNumber(); ++i) {
				ss << "\t";
				info.getDevice(current, i);
				if (defaultDevice.name() == current.name())
					ss << "* ";
				else
					ss << "  ";
				ss << info.version() << " = " << info.name() << endl;
				ss << "\t\t  GL sharing: "
						<< (current.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false")
						<< endl;
				ss << "\t\t  VAAPI media sharing: "
						<< (current.isExtensionSupported("cl_intel_va_api_media_sharing") ?
								"true" : "false") << endl;
			}
		}
    }
#endif
    return ss.str();
}

bool is_intel_va_supported() {
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL()) {
		try {
			std::vector<cv::ocl::PlatformInfo> plt_info;
			cv::ocl::getPlatfomsInfo(plt_info);
			cv::ocl::Device current;
			for (const auto& info : plt_info) {
				for (int i = 0; i < info.deviceNumber(); ++i) {
					info.getDevice(current, i);
					return current.isExtensionSupported("cl_intel_va_api_media_sharing");
				}
			}
		} catch (std::exception& ex) {
			cerr << "Intel VAAPI query failed: " << ex.what() << endl;
		} catch (...) {
			cerr << "Intel VAAPI query failed" << endl;
		}
	}
#endif
    return false;
}

bool is_clgl_sharing_supported() {
#ifdef HAVE_OPENCL
	if(cv::ocl::haveOpenCL()) {
		try {
			if(!cv::ocl::useOpenCL())
				return false;
			std::vector<cv::ocl::PlatformInfo> plt_info;
			cv::ocl::getPlatfomsInfo(plt_info);
			cv::ocl::Device current;
			for (const auto& info : plt_info) {
				for (int i = 0; i < info.deviceNumber(); ++i) {
					info.getDevice(current, i);
					return current.isExtensionSupported("cl_khr_gl_sharing");
				}
			}
		} catch (std::exception& ex) {
			cerr << "CL-GL sharing query failed: " << ex.what() << endl;
		} catch (...) {
			cerr << "CL-GL sharing query failed with unknown error." << endl;
		}
	}
#endif
    return false;
}
static std::mutex finish_mtx;
/*!
 * Internal variable that signals that finishing all operation is requested
 */
static bool finish_requested = false;
/*!
 * Internal variable that tracks if signal handlers have already been installed
 */
static bool signal_handlers_installed = false;

/*!
 * Signal handler callback that signals the application to terminate.
 * @param ignore We ignore the signal number
 */
static void request_finish(int ignore) {
	std::lock_guard guard(finish_mtx);
    CV_UNUSED(ignore);
    finish_requested = true;
}

/*!
 * Installs #request_finish() as signal handler for SIGINT and SIGTERM
 */
static void install_signal_handlers() {
    signal(SIGINT, request_finish);
    signal(SIGTERM, request_finish);
}

bool keep_running() {
	std::lock_guard guard(finish_mtx);
    if (!signal_handlers_installed) {
        install_signal_handlers();
    }
    return !finish_requested;
}

void request_finish() {
	request_finish(0);
}

float aspect_preserving_scale(const cv::Size& scaled, const cv::Size& unscaled) {
	double scale;
	double scaleX =  double(scaled.width) / unscaled.width;
	double scaleY = double(scaled.height) / unscaled.height;
	scale = std::max(scaleX, scaleY);
	if(scale <= 1.0)
		scale = std::min(scaleX, scaleY);
	return scale;
}

void resize_preserving_aspect_ratio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize, const cv::Scalar& bgcolor) {
    cv::UMat tmp;

    double f = aspect_preserving_scale(dstSize, src.size());
    cv::resize(src, tmp, cv::Size(), f, f);

    int top = (dstSize.height - tmp.rows) / 2;
    int down = (dstSize.height - tmp.rows + 1) / 2;
    int left = (dstSize.width - tmp.cols) / 2;
    int right = (dstSize.width - tmp.cols + 1) / 2;

    cv::copyMakeBorder(tmp, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);
}

}
}

