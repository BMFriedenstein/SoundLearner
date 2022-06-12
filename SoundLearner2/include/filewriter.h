/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * filewriter.h
 *  Created on: 03 June 2020
 *      Author: Brandon
 */
#pragma once
#ifndef INCLUDE_FILEWRITER_H_
#define INCLUDE_FILEWRITER_H_

#include <cmath>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "include/common.h"
#include "include/structures.h"

namespace filewriter {
namespace text {
static inline void CreateEmptyFile(const std::string &filename) {
  std::ofstream file(filename, std::ofstream::out);
  file << "";
  file.close();
}

static inline void WriteLine(const std::string &filename, const std::string &line) {
  std::ofstream out_stream(filename, std::ios::app);
  out_stream << line + "\n";
  out_stream.close();
}

static inline void WriteFile(const std::string &filename, const std::string &content) {
  std::ofstream file(filename, std::ofstream::out);
  file << content + "\n";
  file.close();
}
} // namespace text

template <typename T, std::size_t W, std::size_t H> class ImgWriter {
  static_assert(W > 0 && H > 0, "Invalid Image size");

protected:
  std::array<std::array<T, W>, H> img_data;

public:
  explicit ImgWriter(const std::array<std::array<T, W>, H> &a_data) : img_data(a_data) {}
  explicit ImgWriter(std::array<std::array<T, W>, H> &&a_data) : img_data(std::move(a_data)) {}
  virtual ~ImgWriter() {}

  virtual void Write(const std::string &a_file_name) = 0;
};

template <typename T> static inline RGBA ToRgba(T val, ColorScaleType type) {
  static_assert(std::is_arithmetic<T>::value, "Invalid type");
  auto to_uint8 = [](const double &in) -> uint8_t { return static_cast<uint8_t>(std::round(in)); };
   const double clamped_val =
                                              std::clamp(static_cast<double>(val) / std::numeric_limits<T>::max(), 0.0, 1.0);
  switch (type) {
  case ColorScaleType::RGB: {
    const uint32_t val_u32 = clamped_val * 0xFFFFFFU;
    const uint8_t r_val = static_cast<uint8_t>((val_u32 >> 0) & 0XFFU);
    const uint8_t g_val = static_cast<uint8_t>((val_u32 >> 8) & 0XFFU);
    const uint8_t b_val = static_cast<uint8_t>((val_u32 >> 16) & 0XFFU);
    return {r_val, g_val, b_val, 255U};
  }
  case ColorScaleType::YUV: {
    uint8_t r_val = 0;
    uint8_t g_val = 0;
    uint8_t b_val = 0;
    if (0.0 <= clamped_val && clamped_val <= 0.125) {
      r_val = 0;
      g_val = 0;
      b_val = to_uint8((4 * clamped_val + 0.5) * 255);
    } else if (0.125 < clamped_val && clamped_val <= 0.375) {
      r_val = 0;
      g_val = to_uint8((4 * clamped_val - 0.5) * 255);
      b_val = 1;
    } else if (0.375 < clamped_val && clamped_val <= 0.625) {
      r_val = to_uint8((4 * clamped_val - 1.5) * 255);
      g_val = 255;
      b_val = to_uint8((-4 * clamped_val + 2.5) * 255);
    } else if (0.625 < clamped_val && clamped_val <= 0.875) {
      r_val = 255;
      g_val = to_uint8((-4 * clamped_val + 3.5) * 255);
      b_val = 0;
    } else if (0.875 < clamped_val && clamped_val <= 1.0) {
      r_val = to_uint8((-4.0 * clamped_val + 4.5) * 255);
      g_val = 0;
      b_val = 0;
    } else {
      r_val = 255;
      g_val = 0;
      b_val = 0;
    }
    return {r_val, g_val, b_val, 255U};
  }
  case ColorScaleType::GRAYSCALE:
  default: {
    const uint8_t c_val = to_uint8(clamped_val * 255.0);
    return {c_val, c_val, c_val, 255U};
  }
  }
}

namespace ppm {

template <typename T, std::size_t W, std::size_t H, ColorScaleType C = ColorScaleType::GRAYSCALE> class PPMWriter : public ImgWriter<T, W, H> {
public:
  explicit PPMWriter(const std::array<std::array<T, W>, H> &a_data) : ImgWriter<T, W, H>(a_data) {}
  explicit PPMWriter(const std::array<std::array<T, W>, H> &&a_data) : ImgWriter<T, W, H>(std::move(a_data)) {}

  void Write(const std::string &a_file_name) override {
    std::ofstream fout(a_file_name);
    fout << "P3\n" << W << ' ' << H << "\n255\n";
    for (std::size_t j = H - 1; j; --j) {
      for (std::size_t i = 0; i < W; ++i) {
        const RGBA rgba = ToRgba(this->img_data[i][j], C);
        fout << std::to_string(rgba.rgba_st.R) << ' ' << std::to_string(rgba.rgba_st.G) << ' ' << std::to_string(rgba.rgba_st.B) << '\n';
      }
    }
    fout.close();
  }
};
} // namespace ppm

namespace bmp {

template <typename T, std::size_t W, std::size_t H> class BMPWriter : public ImgWriter<T, W, H> {
public:
  explicit BMPWriter(const std::array<std::array<T, W>, H> &a_data) : ImgWriter<T, W, H>(a_data) {
    fileheader.offset_data = sizeof(fileheader.offset_data) + sizeof(infoheader);
    fileheader.file_size = fileheader.offset_data + W * H * sizeof(RGBA) + sizeof(colorheader);
    infoheader.size = sizeof(infoheader);
    infoheader.width = W;
    infoheader.height = H;
    infoheader.compression = 0;
    infoheader.bit_count = 32;
  }

  explicit BMPWriter(const std::array<std::array<T, W>, H> &&a_data) : ImgWriter<T, W, H>(std::move(a_data)) {
    fileheader.offset_data = sizeof(fileheader.offset_data) + sizeof(infoheader);
    fileheader.file_size = fileheader.offset_data + W * H * sizeof(RGBA) + sizeof(colorheader);
    infoheader.size = sizeof(infoheader);
    infoheader.width = W;
    infoheader.height = H;
    infoheader.compression = 0;
    infoheader.bit_count = 32;
  }

  template <ColorScaleType C = ColorScaleType::GRAYSCALE> void Write(const std::string &a_file_name) {
    std::array<RGBA, W * H> flattened_data;
    for (std::size_t i = W - 1; i > 0; --i) {
      for (std::size_t j = H - 1; j > 0; --j) {
        flattened_data[i * H + j] = ToRgba(this->img_data[j][i], C);
      }
    }
    std::fstream fout;
    fout = std::fstream(a_file_name, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&fileheader), sizeof(fileheader));
    fout.write(reinterpret_cast<const char *>(&infoheader), sizeof(infoheader));
    fout.write(reinterpret_cast<const char *>(flattened_data.data()), flattened_data.size() * sizeof(RGBA));
    fout.write(reinterpret_cast<const char *>(&colorheader), sizeof(colorheader));
    fout.close();
  }

  void Write(const std::string &a_file_name) override { Write<ColorScaleType::GRAYSCALE>(a_file_name); }

private:
  BMPFileHeader fileheader;
  BMPInfoHeader infoheader;
  BMPColorHeader colorheader;
};
} // namespace bmp

namespace wave {
class MonoWriter {
public:
  explicit MonoWriter(const std::vector<int16_t> &a_data) {
    wav_data.resize(a_data.size() * sizeof(int16_t));
    std::memcpy(wav_data.data(), a_data.data(), wav_data.size());
    header.chunk_size = wav_data.size() + 36;
    header.sub_chunk_1_size = 16;
    header.audio_format = 1;
    header.num_of_channels = 1;
    header.sample_rate = 44100;
    header.bytes_per_second = 88200;
    header.block_allign = 2;
    header.bit_depth = 16;
    header.sub_chunk_2_size = wav_data.size();
  }

  void Write(const std::string &a_file_name) {
    std::vector<char> data(wav_data.size() + sizeof(WavFileHeader));
    memcpy(data.data(), reinterpret_cast<const uint8_t *>(&header), sizeof(WavFileHeader));
    memcpy(data.data() + sizeof(WavFileHeader), wav_data.data(), wav_data.size());
    std::ofstream fout(a_file_name, std::ios::out | std::ios::binary);
    fout.write(data.data(), data.size());
    fout.close();
  }

private:
  WavFileHeader header;
  std::vector<char> wav_data;
};
} // namespace wave
} // namespace filewriter

#endif // INCLUDE_FILEWRITER_H_
