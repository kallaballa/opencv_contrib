For videos & screenshots, progress and more see: https://github.com/opencv/opencv/issues/22923

## Introduction to "Plan" and "V4D"

### Overview of "Plan"
**Plan** is a computational graph engine built with C++20 templates, enabling developers to construct directed acyclic graphs (DAGs) from fragments of algorithms. By leveraging these graphs, Plan facilitates the optimization of parallel and concurrent algorithms, ensuring efficient resource utilization. The framework divides the lifetime of an algorithm into two distinct phases: **inference** and **execution**. 

- **Inference Phase:** During this phase, the computational graph is constructed by running the Plan implementation. This process organizes the algorithm's fragments and binds them to data, which may be classified as:
  - **Safe Data:** Member variables of the Plan.
  - **Shared Data:** External variables (e.g., global or static data).
  
  Functions and data are explicitly flagged as shared when necessary, adhering to Plan’s transparent approach to state management. The framework discourages hidden states, as they impede program integrity and graph optimization. 

- **Execution Phase:** This phase executes the constructed graph using the defined nodes and edges. Nodes typically represent algorithmic fragments such as functions or lambdas, while edges define data flow, supporting various access patterns (e.g., read, write, copy).

Plan also allows hierarchical composition, where one Plan may be composed of other sub-Plans. Special rules govern data sharing in such compositions to maintain performance and correctness. Currently, optimizations are limited to “best-effort” pipelining, with plans for more sophisticated enhancements.

### Overview of "V4D"
**V4D** is a versatile 2D/3D graphics runtime designed to integrate seamlessly with Plan. Built atop OpenGL (3.0 or ES 3.2), V4D extends its functionality through bindings to prominent libraries:
- **NanoVG:** For 2D vector and raster graphics, including font rendering.
- **bgfx:** A 3D engine modified to defer its concurrency model to Plan for optimal parallel execution.
- **IMGui:** A lightweight GUI overlay.

V4D encourages direct OpenGL usage and external API integrations via **context sharing**, which is implemented using shared textures. Each external API operates within its isolated OpenGL state machine, maintaining thread safety and modularity.

The runtime’s capabilities are further augmented by its integration with OpenCV, providing:
- **Hardware Acceleration:** Utilizing OpenGL for graphics, VAAPI and NVENC for video, and OpenCL-OpenGL interop for compute tasks.
- **Data Sharing on GPU:** Depending on hardware and software features, V4D can directly share or copy data within GPU memory for efficient processing.

### Integration and Platform Support
V4D and Plan share a tightly bonded design, simplifying combined use cases. However, plans are underway to decouple them, enabling the adoption of alternative runtimes. V4D is actively developed for Linux (X11 and Wayland via EGL or GLX), with auto-detection of supported backends. While macOS support lags slightly, Windows compatibility remains untested but is considered during development.

### Key Principles and Features
1. **Fine-Grained Edge Calls:** Plan introduces specialized edge calls (e.g., `R`, `RW`, `V`) to define data access patterns, supporting smart pointers and OpenCV `UMat` objects. This granularity allows better graph optimization.
2. **State and Data Transparency:** Functions and data in a Plan must avoid introducing hidden states unless explicitly marked as shared. This principle ensures the integrity of the graph and its optimizations.
3. **Parallelism and Pipelining:** Multiple OpenGL contexts can be created and utilized in parallel, making V4D a robust solution for high-performance graphics applications.
4. **Algorithm Modularity:** By structuring algorithms into smaller, reusable fragments or sub-Plans, Plan fosters modular development and scalability.

## Selected Commented Examples (read sequentially)
The following examples have been selected to deepen your understanding of Plan-V4D. There are many more.

### Blue Sreen using OpenGL
[source](modules/v4d/samples/render_opengl.cpp)

### Displaying an Image using NanoVG
[source](modules/v4d/samples/display_image_nvg.cpp)

### A realtime beauty filter (using sub-plans)
[source](modules/v4d/samples/beauty-demo.cpp)

## Why Plan-V4D?

* Computational Graph Engine: Fast parallel code.
* OpenGL: Easy access to OpenGL.
* GUI: Simple yet powerful user interfaces through ImGui.
* Vector graphics: Elegant and fast vector graphics through NanoVG.
* 3D graphics: Powerful 3D graphics through bgfx.
* Font rendering: Loading of fonts and sophisticated rendering options.
* Video pipeline: Through a simple source/sink system videos can be efficently read, displayed, edited and saved.
* Hardware acceleration: Transparent hardware acceleration usage where possible. (e.g. OpenGL, OpenCL, CL-GL interop, VAAPI and CL-VAAPI interop, nvenc). Actually it is possible to write programs that 
* No more highgui with it's heavy dependencies, licenses and limitations.

Please refer to the examples and demos as well as [this OpenCV issue](https://github.com/opencv/opencv/issues/22923) to find out exactly what it can do for you.

## Platforms

* Linux
* Mac OS
* Planned but entirely untested: Windows

## GPU Support
* Intel iGPU Gen 8+ (Tested: Gen 11 & Gen 13)
* NVIDIA Ada Lovelace (Tested: GTX 4070 Ti) with proprietary drivers (535.104.05) and CUDA toolkit (12.2) tested.
* Intel Arc770 (Mesa 24.3.1) tested
* AMD: never tested
* Uses software rendering through Mesa for testing as well

## Requirements
* C++20 (at the moment)
* OpenGL 3.2 Core (optionally Compat)/OpenGL ES 3.0/WebGL2

## Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.

## Dependencies
* My OpenCV 4.x fork (It works with mainline OpenCV 4.x as well, but using my fork is highly recommended because it features several improvements and fixes)
* GLEW
* GLFW3
* NanoVG (included as a sub-repo)
* ImGui (included as a sub-repo)
* bgfx (included as a sub-repo)
* Glad (included)

## Instructions for Linux (Ubuntu 24.04.1 LTS - Noble Numbat)

### Optional: Create a chroot if you are using another distribution or Ubuntu version 

#### Build minbase chroot (basically installing a minimal Ubuntu into the directory plan-v4d-noble)
```bash
sudo debootstrap --variant=minbase --arch=amd64 noble plan-v4d-noble http://archive.ubuntu.com/ubuntu/
```

#### Bind /dev - WARNING: Don't delete the chroot without umounting plan-v4d-noble/dev or your system will crash
```bash
sudo mount --bind /dev/ plan-v4d-noble/dev
```

#### Enter the chroot - from here on you are inside the ubuntu chroot and you cannot access the files of your system anymore until you leave it (e.g: by running "exit")
```bash
sudo chroot plan-v4d-noble/
```

#### Setup essential virtual filesystems - NOTE: umount those after you are done with the chroot
```bash
mount -t proc none /proc
mount -t sysfs none /sys
mount -t tmpfs none /tmp
mount -t devpts none /dev/pts
```

### Configure apt
```bash
echo "deb http://archive.ubuntu.com/ubuntu noble main universe" > /etc/apt/sources.list
```

### Install required packages
```bash
apt update
apt install vainfo clinfo libqt5opengl5-dev freeglut3-dev ocl-icd-opencl-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen libxinerama-dev libxcursor-dev libxi-dev libva-dev yt-dlp wget intel-opencl-icd ca-certificates
```

### Optional: install packaging tools
```bash
apt install ubuntu-dev-tools dh-cmake gdebi
```

### EITHER: Minimal Plan-V4D build WITHOUT examples, demos and packages
```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/Plan-V4D.git
mkdir opencv/build
cd opencv/build

# Configuring a Wayland (-DWITH_WAYLAND=ON) build without examples (-DBUILD_EXAMPLES=OFF) and packages (-DBUILD_PACKAGE=OFF)
```bash
cmake -DWITH_WAYLAND=ON -DOPENCV_V4D_ENABLE_ES3=OFF -DCMAKE_CXX_FLAGS="-DCL_TARGET_OPENCL_VERSION=120" -DCMAKE_MODULE_LINKER_FLAGS="/usr/local/lib64/" -DINSTALL_BIN_EXAMPLES=OFF -DOPENCV_CUSTOM_PACKAGE_INFO=ON -DCPACK_PACKAGE_VERSION_MAJOR=4 -DCPACK_PACKAGE_VERSION_MINOR=10 -DCPACK_PACKAGE_VERSION_PATCH=0 -DCPACK_PACKAGE_VERSION=4:10.0-yourname -DCMAKE_BUILD_TYPE=Release -DCPACK_PACKAGE_CONTACT="you@example.com" -DOPENCV_GENERATE_PKGCONFIG=ON -DCPACK_PACKAGE_VENDOR=yourname -DCPACK_DEBIAN_PACKAGE_DEPENDS="libqt5opengl5,freeglut3,ocl-icd-libopencl1,libavcodec58,libavdevice58,libavfilter7,libavformat58,libavutil56,libpostproc55,libswresample3,libswscale5,libglfw3,libstb0,libglew2.2,zlib1g,libxinerama1,libxcursor1,libxi6,libva2,intel-opencl-icd,ca-certificates" -DINSTALL_CREATE_DISTRIB=ON -DCPACK_BINARY_DEB=ON -DCV_TRACE=OFF -DBUILD_SHARED_LIBS=ON -DWITH_OPENGL=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF -DWITH_EIGEN=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_OPENVX=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF-DWITH_TBB=OFF -DWITH_TIFF=OFF -DWITH_OPENCL=ON -DWITH_OPENCL_SVM=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=OFF -DWITH_ITT=OFF -DWITH_QUIRC=ON -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=ON -DBUIlD_opencv_ccalib=OFF -DBUILD_opencv_dnn=ON -DBUILD_opencv_features2d=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_gapi=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=ON -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_shape=OFF -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_highgui=ON -DBUILD_opencv_superres=OFF -DBUILD_opencv_stitching=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_js=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_alphamat=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_barcode=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_cudaarithm=OFF -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_cudafeatures2d=OFF -DBUILD_opencv_cudafilters=OFF -DBUILD_opencv_cudaimgproc=OFF -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_cudaobjdetect=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudastereo=OFF -DBUILD_opencv_cudawarping=OFF -DBUILD_opencv_cudev=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dnn_superres=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_freetype=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF -DBUILD_opencv_intensity_transform=OFF -DBUILD_opencv_julia=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_mcc=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_ovis=OFF -DBUILD_opencv_phase_unwrapping=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_quality=OFF -DBUILD_opencv_rapid=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_shape=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_wechat_qrcode=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DBUILD_opencv_world=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PACKAGE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DWITH_PTHREADS_PF=ON -DCV_ENABLE_INTRINSICS=ON -DBUILD_opencv_video=ON -DBUILD_opencv_v4d=ON -DGBFX_CONFIG_MULTITHREADED=OFF -DBGFX_CONFIG_PASSIVE=ON -DOPENCV_EXTRA_MODULES_PATH="../../Plan-V4D/modules" ..
```
### OR: Full Plan-V4D build WITH examples, demos and packages
```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/Plan-V4D.git
mkdir opencv/build
cd opencv/build

# Configuring a Wayland (-DWITH_WAYLAND=ON) build with examples (-DBUILD_EXAMPLES=ON) and packages (-DBUILD_PACKAGE=ON) - NOTE: you might want to update the package info (-DCPACK_PACKAGE_VERSION, -DCPACK_PACKAGE_CONTACT, -DCPACK_PACKAGE_VENDOR)
```bash
cmake -DWITH_WAYLAND=ON -DOPENCV_V4D_ENABLE_ES3=OFF -DCMAKE_CXX_FLAGS="-DCL_TARGET_OPENCL_VERSION=120" -DCMAKE_MODULE_LINKER_FLAGS="/usr/local/lib64/" -DINSTALL_BIN_EXAMPLES=OFF -DOPENCV_CUSTOM_PACKAGE_INFO=ON -DCPACK_PACKAGE_VERSION_MAJOR=4 -DCPACK_PACKAGE_VERSION_MINOR=10 -DCPACK_PACKAGE_VERSION_PATCH=0 -DCPACK_PACKAGE_VERSION=4:10.0-yourname -DCMAKE_BUILD_TYPE=Release -DCPACK_PACKAGE_CONTACT="you@example.com" -DOPENCV_GENERATE_PKGCONFIG=ON -DCPACK_PACKAGE_VENDOR=yourname -DCPACK_DEBIAN_PACKAGE_DEPENDS="libqt5opengl5,freeglut3,ocl-icd-libopencl1,libavcodec58,libavdevice58,libavfilter7,libavformat58,libavutil56,libpostproc55,libswresample3,libswscale5,libglfw3,libstb0,libglew2.2,zlib1g,libxinerama1,libxcursor1,libxi6,libva2,intel-opencl-icd,ca-certificates" -DINSTALL_CREATE_DISTRIB=ON -DCPACK_BINARY_DEB=ON -DCV_TRACE=OFF -DBUILD_SHARED_LIBS=ON -DWITH_OPENGL=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF -DWITH_EIGEN=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_OPENVX=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF-DWITH_TBB=OFF -DWITH_TIFF=OFF -DWITH_OPENCL=ON -DWITH_OPENCL_SVM=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=OFF -DWITH_ITT=OFF -DWITH_QUIRC=ON -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=ON -DBUIlD_opencv_ccalib=OFF -DBUILD_opencv_dnn=ON -DBUILD_opencv_features2d=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_gapi=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=ON -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_shape=OFF -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_highgui=ON -DBUILD_opencv_superres=OFF -DBUILD_opencv_stitching=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_js=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_alphamat=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_barcode=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_cudaarithm=OFF -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_cudafeatures2d=OFF -DBUILD_opencv_cudafilters=OFF -DBUILD_opencv_cudaimgproc=OFF -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_cudaobjdetect=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudastereo=OFF -DBUILD_opencv_cudawarping=OFF -DBUILD_opencv_cudev=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dnn_superres=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_freetype=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF -DBUILD_opencv_intensity_transform=OFF -DBUILD_opencv_julia=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_mcc=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_ovis=OFF -DBUILD_opencv_phase_unwrapping=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_quality=OFF -DBUILD_opencv_rapid=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_shape=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_wechat_qrcode=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DBUILD_opencv_world=OFF -DBUILD_EXAMPLES=ON -DBUILD_PACKAGE=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DWITH_PTHREADS_PF=ON -DCV_ENABLE_INTRINSICS=ON -DBUILD_opencv_video=ON -DBUILD_opencv_v4d=ON -DGBFX_CONFIG_MULTITHREADED=OFF -DBGFX_CONFIG_PASSIVE=ON -DOPENCV_EXTRA_MODULES_PATH="../../Plan-V4D/modules" ..
```

### Build
```bash
make -j8
```

### Optional: Build packages
```bash
cpack DEB
```

### Cleaning up and leaving the chroot
```bash
umount proc
umount sys
umount tmp
umount dev/pts
exit #After this command you are back outside the chroot with access to your system.
```

### Unbind dev
```bash
sudo umount plan-v4d-noble/dev
```

## Instructions for Mac OS X Using Homebrew

### Install Required Dependencies
First, ensure you have Homebrew installed on your system. Then, install the required dependencies:

```bash
# Update Homebrew
brew update

# Install packages
brew install cmake git qt@5 glfw glew zlib doxygen wget yt-dlp
brew install clinfo # For OpenCL information
brew install opencl-headers # OpenCL headers
```

### Clone Repositories
```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/Plan-V4D.git
mkdir opencv/build
cd opencv/build
```

### Minimal Plan-V4D Build (without examples, demos, and packages)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_V4D_ENABLE_ES3=OFF \
      -DWITH_QT=ON \
      -DWITH_OPENGL=ON \
      -DWITH_FFMPEG=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_PACKAGE=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DWITH_OPENCL=ON \
      -DOPENCV_EXTRA_MODULES_PATH="../../Plan-V4D/modules" ..
```

### Full Plan-V4D Build (with examples, demos, and packages)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_V4D_ENABLE_ES3=OFF \
      -DWITH_QT=ON \
      -DWITH_OPENGL=ON \
      -DWITH_FFMPEG=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_PACKAGE=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DWITH_OPENCL=ON \
      -DOPENCV_EXTRA_MODULES_PATH="../../Plan-V4D/modules" ..
```

### Build the Project
```bash
make -j$(sysctl -n hw.ncpu)
```

## Download the example videos
```bash
# big buck bunny video
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
# dance video
yt-dlp -o dance.webm "https://www.youtube.com/watch?v=yg6LZtNeO_8"
# kristen video
yt-dlp -o kristen.webm "https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s"
```

## Run the examples and demos
```bash
# Examples
bin/example_v4d_display_image
bin/example_v4d_display_image_fb
bin/example_v4d_vector_graphics
bin/example_v4d_vector_graphics_and_fb
bin/example_v4d_render_opengl
bin/example_v4d_font_rendering
bin/example_v4d_video_editing
bin/example_v4d_custom_source_and_sink
bin/example_v4d_font_with_gui
 
# Demos
bin/example_v4d_cube-demo
bin/example_v4d_many_cubes-demo
bin/example_v4d_video-demo bunny.webm
bin/example_v4d_nanovg-demo bunny.webm
bin/example_v4d_shader-demo bunny.webm
bin/example_v4d_font-demo
bin/example_v4d_pedestrian-demo dance.webm
bin/example_v4d_optflow-demo dance.webm
bin/example_v4d_beauty-demo kristen.webm
```

## Attribution
* The author of the bunny video is the Blender Foundation ([Original video](https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm)).
* The author of the dance video is GNI Dance Company ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8)).
* The author of the video used in the beauty-demo video is Kristen Leanne ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw)).
* The author of cxxpool is Copyright (c) 2022 Christian Blume: ([LICENSE](https://github.com/bloomen/cxxpool/blob/master/LICENSE))
* The author of the roboto font family is Google Inc. ([LICENSE](https://github.com/googlefonts/roboto/blob/main/LICENSE))
