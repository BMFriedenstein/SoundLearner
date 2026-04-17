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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "include/common.h"
#include "include/structures.h"

namespace filewriter {

namespace detail {

static inline void EnsureOpen(const std::ofstream &stream, const std::string &filename) {
  if (!stream) {
    throw std::runtime_error("Unable to open output file: " + filename);
  }
}

template <typename T> static inline double NormalizePixel(T value) {
  static_assert(std::is_arithmetic_v<T>, "Image pixels must be arithmetic values");

  if constexpr (std::is_floating_point_v<T>) {
    return std::clamp(static_cast<double>(value), 0.0, 1.0);
  } else {
    constexpr auto max_value = std::numeric_limits<T>::max();
    if constexpr (max_value == 0) {
      return 0.0;
    } else {
      return std::clamp(static_cast<double>(value) / static_cast<double>(max_value), 0.0, 1.0);
    }
  }
}

static inline uint8_t ToByte(double value) {
  return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
}

static inline RGBA MakeRgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255U) {
  RGBA rgba{};
  rgba.rgba_st.R = r;
  rgba.rgba_st.G = g;
  rgba.rgba_st.B = b;
  rgba.rgba_st.A = a;
  return rgba;
}

} // namespace detail

namespace text {

static inline void CreateEmptyFile(const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  detail::EnsureOpen(file, filename);
}

static inline void WriteLine(const std::string &filename, const std::string &line) {
  std::ofstream out_stream(filename, std::ios::out | std::ios::app);
  detail::EnsureOpen(out_stream, filename);
  out_stream << line << '\n';
}

static inline void WriteFile(const std::string &filename, const std::string &content) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  detail::EnsureOpen(file, filename);
  file << content << '\n';
}

} // namespace text

template <typename T, std::size_t W, std::size_t H> class ImgWriter {
  static_assert(W > 0 && H > 0, "Invalid image size");

protected:
  using ImageData = std::array<std::array<T, W>, H>;

  std::unique_ptr<ImageData> owned_img_data;
  const ImageData *img_data;

public:
  explicit ImgWriter(const ImageData &data) : img_data(&data) {}
  explicit ImgWriter(ImageData &&data) : owned_img_data(std::make_unique<ImageData>(std::move(data))), img_data(owned_img_data.get()) {}
  virtual ~ImgWriter() = default;

  virtual void Write(const std::string &file_name) = 0;
};

template <typename T> static inline RGBA ToRgba(T value, ColorScaleType type) {
  const double clamped_val = detail::NormalizePixel(value);

  switch (type) {
  case ColorScaleType::RGB: {
    const uint8_t r_val = detail::ToByte(clamped_val);
    const uint8_t g_val = detail::ToByte(std::sqrt(clamped_val));
    const uint8_t b_val = detail::ToByte(1.0 - clamped_val);
    return detail::MakeRgba(r_val, g_val, b_val);
  }
  case ColorScaleType::YUV: {
    uint8_t r_val = 0;
    uint8_t g_val = 0;
    uint8_t b_val = 0;
    if (clamped_val <= 0.125) {
      b_val = detail::ToByte(4.0 * clamped_val + 0.5);
    } else if (clamped_val <= 0.375) {
      g_val = detail::ToByte(4.0 * clamped_val - 0.5);
      b_val = 255;
    } else if (clamped_val <= 0.625) {
      r_val = detail::ToByte(4.0 * clamped_val - 1.5);
      g_val = 255;
      b_val = detail::ToByte(-4.0 * clamped_val + 2.5);
    } else if (clamped_val <= 0.875) {
      r_val = 255;
      g_val = detail::ToByte(-4.0 * clamped_val + 3.5);
    } else {
      r_val = detail::ToByte(-4.0 * clamped_val + 4.5);
    }
    return detail::MakeRgba(r_val, g_val, b_val);
  }
  case ColorScaleType::GRAYSCALE:
  default: {
    const uint8_t c_val = detail::ToByte(clamped_val);
    return detail::MakeRgba(c_val, c_val, c_val);
  }
  }
}

namespace ppm {

template <typename T, std::size_t W, std::size_t H, ColorScaleType C = ColorScaleType::GRAYSCALE> class PPMWriter : public ImgWriter<T, W, H> {
public:
  using ImageData = typename ImgWriter<T, W, H>::ImageData;

  explicit PPMWriter(const ImageData &data) : ImgWriter<T, W, H>(data) {}
  explicit PPMWriter(ImageData &&data) : ImgWriter<T, W, H>(std::move(data)) {}

  void Write(const std::string &file_name) override {
    std::ofstream fout(file_name, std::ios::out | std::ios::trunc);
    detail::EnsureOpen(fout, file_name);

    fout << "P3\n" << W << ' ' << H << "\n255\n";
    for (std::size_t row = H; row-- > 0;) {
      for (std::size_t col = 0; col < W; ++col) {
        const RGBA rgba = ToRgba((*this->img_data)[col][row], C);
        fout << static_cast<int>(rgba.rgba_st.R) << ' ' << static_cast<int>(rgba.rgba_st.G) << ' ' << static_cast<int>(rgba.rgba_st.B) << '\n';
      }
    }
  }
};

} // namespace ppm

namespace bmp {

template <typename T, std::size_t W, std::size_t H> class BMPWriter : public ImgWriter<T, W, H> {
public:
  using ImageData = typename ImgWriter<T, W, H>::ImageData;

  explicit BMPWriter(const ImageData &data) : ImgWriter<T, W, H>(data) {}
  explicit BMPWriter(ImageData &&data) : ImgWriter<T, W, H>(std::move(data)) {}

  template <ColorScaleType C = ColorScaleType::GRAYSCALE> void Write(const std::string &file_name) {
    BMPFileHeader file_header{};
    BMPInfoHeader info_header{};

    info_header.size = sizeof(BMPInfoHeader);
    info_header.width = static_cast<int32_t>(W);
    info_header.height = static_cast<int32_t>(H);
    info_header.bit_count = 32;
    info_header.compression = 0;
    info_header.size_image = static_cast<uint32_t>(W * H * sizeof(RGBA));

    file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    file_header.file_size = file_header.offset_data + info_header.size_image;

    std::ofstream fout(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    detail::EnsureOpen(fout, file_name);
    fout.write(reinterpret_cast<const char *>(&file_header), sizeof(file_header));
    fout.write(reinterpret_cast<const char *>(&info_header), sizeof(info_header));

    for (std::size_t row = H; row-- > 0;) {
      for (std::size_t col = 0; col < W; ++col) {
        const RGBA rgba = ToRgba((*this->img_data)[col][row], C);
        fout.write(reinterpret_cast<const char *>(&rgba), sizeof(rgba));
      }
    }
  }

  void Write(const std::string &file_name) override { Write<ColorScaleType::GRAYSCALE>(file_name); }
};

} // namespace bmp

namespace wave {

class MonoWriter {
public:
  explicit MonoWriter(const std::vector<int16_t> &data) : wav_data(data) {}

  void Write(const std::string &file_name) {
    WavFileHeader header{};
    header.num_of_channels = 1;
    header.sample_rate = SAMPLE_RATE;
    header.bit_depth = BITDEPTH;
    header.block_allign = static_cast<uint16_t>(header.num_of_channels * header.bit_depth / 8);
    header.bytes_per_second = header.sample_rate * header.block_allign;
    header.sub_chunk_2_size = static_cast<uint32_t>(wav_data.size() * sizeof(int16_t));
    header.chunk_size = 36U + header.sub_chunk_2_size;

    std::ofstream fout(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    detail::EnsureOpen(fout, file_name);
    fout.write(reinterpret_cast<const char *>(&header), sizeof(header));
    fout.write(reinterpret_cast<const char *>(wav_data.data()), static_cast<std::streamsize>(header.sub_chunk_2_size));
  }

private:
  std::vector<int16_t> wav_data;
};

} // namespace wave
} // namespace filewriter

#endif // INCLUDE_FILEWRITER_H_
