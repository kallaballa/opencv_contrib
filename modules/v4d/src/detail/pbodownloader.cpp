#if defined(__linux)
#  include <string.h>
#endif
#include <iostream>
#include <cassert>
#include <chrono>
#include "pbodownloader.hpp"

namespace poly {
#ifdef OPENCV_V4D_USE_ES3
#define GL_BGRA GL_RGBA
#define GL_BGR GL_RGB
#endif

#define SX_ERROR_(msg) printf("%s\n", msg);
#define SX_WARNING_(msg) printf("%s\n", msg);
#define SX_DEBUG_(msg) printf("%s\n", msg);
#define SX_VERBOSE_(msg) printf("%s\n", msg);
#define SX_ERROR(fmt, ...) printf(fmt, __VA_ARGS__);
#define SX_WARNING(fmt, ...) printf(fmt, __VA_ARGS__);
#define SX_DEBUG(fmt, ...) printf(fmt, __VA_ARGS__);
#define SX_VERBOSE(fmt, ...) printf(fmt, __VA_ARGS__);

void myGetBufferSubData (GLenum theTarget, GLintptr theOffset, GLsizeiptr theSize, void* theData)
{
#ifdef __EMSCRIPTEN__
  EM_ASM_(
  {
    Module.ctx.getBufferSubData($0, $1, HEAPU8.subarray($2, $2 + $3));
  }, theTarget, theOffset, theData, theSize);
#else
  glGetBufferSubData (theTarget, theOffset, theSize, theData);
#endif
}

  PboDownloader::PboDownloader()
    :fmt(0)
    ,pbos(NULL)
    ,num_pbos(0)
    ,dx(0)
    ,num_downloads(0)
    ,width(0)
    ,height(0)
    ,nbytes(0)
    ,pixels(NULL)
  {
  }

  PboDownloader::~PboDownloader() {
    if (NULL != pixels) {
      delete[] pixels;
      pixels = NULL;
    }
  }

  int PboDownloader::init(GLenum format, int w, int h, int num) {

    if (NULL != pbos) {
      SX_ERROR_("Already initialized. Not necessary to initialize again; or shutdown first.");
      return -1;
    }

    if (0 >= num) {
      SX_ERROR("Invalid number of PBOs: %d\n", num);
      return -2;
    }

    if (num > 10) {
      SX_WARNING_("Asked to create more then 10 buffers; that is probaly a bit too much.");
    }

    fmt = format;
    width = w;
    height = h;
    num_pbos = num;

    if (GL_RED == fmt || GL_GREEN == fmt || GL_BLUE == fmt) {
      nbytes = width * height;
    }
    else if (GL_RGB == fmt || GL_BGR == fmt) {
      nbytes = width * height * 3;
    }
    else if (GL_RGBA == fmt || GL_BGRA == fmt) {
      nbytes = width * height * 4;
    }
    else {
      SX_ERROR_("Unhandled pixel format, use GL_R, GL_RG, GL_RGB or GL_RGBA.");
      return -3;
    }

    if (0 == nbytes) {
      SX_ERROR("Invalid width or height given: %d x %d\n", width, height);
      return -4;
    }

    pbos = new GLuint[num];
    if (NULL == pbos) {
      SX_ERROR_("Cannot allocate pbos.");
      return -3;
    }

    pixels = new unsigned char[nbytes];
    if (NULL == pixels) {
      SX_ERROR_("Cannot allocate pixel buffer.");
      return -5;
    }

    glGenBuffers(num, pbos);
    for (int i = 0; i < num; ++i) {

      SX_VERBOSE("pbodownloader.pbos[%d] = %d, nbytes: %d\n", i, pbos[i], nbytes)

      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
      glBufferData(GL_PIXEL_PACK_BUFFER, nbytes, NULL, GL_STREAM_READ);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    return 0;
  }

  uint64_t nanos() {
      return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
  void PboDownloader::download() {
    unsigned char* ptr;
    uint64_t start_ns = nanos();
    uint64_t end_ns = 0;
    uint64_t delta_ns = 0;

#define USE_PBO 1
#if USE_PBO
    cerr << "1" << endl;
    if (num_downloads < num_pbos) {
        cerr << "2" << endl;
      /*
         First we need to make sure all our pbos are bound, so glMap/Unmap will
         read from the oldest bound buffer first.
      */
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[dx]);
      glReadPixels(0, 0, width, height, fmt, GL_UNSIGNED_BYTE, 0);   /* When a GL_PIXEL_PACK_BUFFER is bound, the last 0 is used as offset into the buffer to read into. */
      SX_DEBUG("glReadPixels() with pbo: %d\n", pbos[dx]);
    }
    else {
        cerr << "3" << endl;
      SX_DEBUG("glMapBuffer() with pbo: %d\n", pbos[dx]);

      /* Read from the oldest bound pbo. */
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[dx]);
#ifdef __EMSCRIPTEN__
      std::unique_ptr<uint8_t> clientBuffer = std::make_unique<uint8_t>(nbytes);
      myGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, nbytes, clientBuffer.get());
      ptr = clientBuffer.get();
#else
      ptr = (unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
#endif
      if (NULL != ptr) {
        memcpy(pixels, ptr, nbytes);
#ifndef __EMSCRIPTEN__
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
#endif
      }
      else {
          cerr << "4" << endl;
        SX_ERROR_("Failed to map the buffer\n");
      }

      /* Trigger the next read. */
      SX_DEBUG("glReadPixels() with pbo: %d\n", pbos[dx]);
      glReadPixels(0, 0, width, height, fmt, GL_UNSIGNED_BYTE, 0);
    }

    ++dx;
    dx = dx % num_pbos;

    num_downloads++;
    if (num_downloads == UINT64_MAX) {
      num_downloads = num_pbos;
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
#else
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); /* just make sure we're not accidentilly using a PBO. */
    glReadPixels(0, 0, width, height, fmt, GL_UNSIGNED_BYTE, pixels);
#endif

    end_ns = nanos();

    delta_ns = end_ns - start_ns;
    SX_VERBOSE("Download took: %f ms. \n", ((double)delta_ns) / 1000000.0);
  }

} /* namespace poly */
