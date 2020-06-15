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
#include <cstring>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include "structures.h"

namespace filewriter {
namespace text {
static inline void CreateEmptyFile(const std::string& filename) {
  std::fstream file(filename, std::fstream::out);
  file << "";
  file.close();
}

static inline void WriteLine(const std::string& filename, const std::string& line) {
  std::ofstream out_stream(filename, std::ios::app);
  out_stream << line + "\n";
  out_stream.close();
}

static inline void WriteFile(const std::string& filename, const std::string& content) {
  std::fstream file(filename, std::fstream::out);
  file << content + "\n";
  file.close();
}
}  // namespace text

namespace bmp {

template <std::size_t W, std::size_t H>
class BMPWriterC {
 public:
  BMPWriterC(const std::array<std::array<uint32_t, W>, H>& a_data) {
    for (size_t i = H - 1; i > 0; --i) {
      for (size_t j = W - 1; j > 0; --j) {
        img_data[i * W + j] = ValToColor(double(a_data[j][i]));
      }
    }
    fileheader.offset_data = sizeof(fileheader) + sizeof(infoheader);
    fileheader.file_size = fileheader.offset_data + img_data.size() * 4 + sizeof(colorheader);
    infoheader.size = sizeof(infoheader);
    infoheader.width = W;
    infoheader.height = H;
    infoheader.compression = 0;
    infoheader.bit_count = 32;
  }

  BMPWriterC(const std::array<std::array<double, W>, H>& a_data) {
    for (size_t i = W - 1; i > 0; --i) {
      for (size_t j = H - 1; j > 0; --j) {
        img_data[i * H + j] = ValToColor(double(a_data[j][i]));
      }
    }
    fileheader.offset_data = sizeof(fileheader.offset_data) + sizeof(infoheader);
    fileheader.file_size = fileheader.offset_data + img_data.size() * 4 + sizeof(colorheader);

    infoheader.size = sizeof(infoheader);
    infoheader.width = W;
    infoheader.height = H;
    infoheader.compression = 0;
    infoheader.bit_count = 32;
  }

  void Write(const std::string& a_file_name) {
    if (W == 0 || H == 0) {
      std::cout << "WARNING, cannot write img as W || H is 0";
      return;
    }

    std::fstream fout;
    fout = std::fstream(a_file_name, std::ios::out | std::ios::binary);
    fout.write((const char*)&fileheader, sizeof(fileheader));
    fout.write((const char*)&infoheader, sizeof(infoheader));
    fout.write((const char*)img_data.data(), img_data.size() * 4);
    fout.write((const char*)&colorheader, sizeof(colorheader));
    fout.close();
  }

 private:
  enum ColorScaleType { GREYSCALE = 0 };
  uint32_t ValToColor(double val, ColorScaleType type = GREYSCALE) {
    const uint8_t val_u = std::clamp<uint8_t>(static_cast<uint8_t>(std::round(0xff * val)), 0, 0xff);

    switch (type) {
      case GREYSCALE:
        RGBA val_rgba;
        val_rgba.rgba_st.A = 0xff;
        val_rgba.rgba_st.R = val_u;
        val_rgba.rgba_st.G = val_u;
        val_rgba.rgba_st.B = val_u;
        return val_rgba.rgba;
      default:
        return 0xffffffff;
    }
  }
  std::array<unsigned char, 3> bmppad;
  BMPFileHeader fileheader;
  BMPInfoHeader infoheader;
  BMPColorHeader colorheader;
  std::array<uint32_t, W * H> img_data;
};
}  // namespace bmp

namespace wave {
class MonoWaveWriterC {
 public:
  explicit MonoWaveWriterC(const std::vector<int16_t>& a_data) {
    wav_data.resize(a_data.size() * 2);
    std::memcpy(reinterpret_cast<char*>(wav_data.data()), reinterpret_cast<const char*>(a_data.data()),
                wav_data.size());
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

  void Write(const std::string& a_file_name) {
    std::vector<char> data(wav_data.size() + sizeof(WavFileHeader));
    memcpy(data.data(), &header, sizeof(WavFileHeader));
    memcpy(data.data() + sizeof(WavFileHeader), wav_data.data(), wav_data.size());
    std::ofstream fout(a_file_name, std::ios::out | std::ios::binary);
    fout.write(data.data(), data.size());
    fout.close();
  }

 private:
  WavFileHeader header;
  std::vector<char> wav_data;
};
}  // namespace wave
}  // namespace filewriter

#endif  // INCLUDE_FILEWRITER_H_