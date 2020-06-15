/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * filereader.h
 *  Created on: 03 June 2020
 *      Author: Brandon
 */
#pragma once
#ifndef INCLUDE_FILEREADER_H_
#define INCLUDE_FILEREADER_H_
#include <cstring>

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/common.h"
#include "include/structures.h"

namespace filereader {

namespace bmp {
class BMPReaderC {
 public:
  explicit BMPReaderC(const std::string& a_filename);
  std::string HeaderToString();

 private:
  std::string filename;
  WavFileHeader header;
  std::vector<char> wav_data;
};
}  // namespace bmp

namespace wave {
class WaveReaderC {
 public:
  explicit WaveReaderC(const std::string& a_filename) : filename(a_filename) {
    // Open file.
    std::ifstream wav_file(a_filename, std::ios::binary);
    if (!wav_file.is_open()) {
      std::cout << "ERROR!!! Could not open wave file " << a_filename << std::endl;
      exit(EXIT_READ_FILE_FAILED);
    }
    wav_file.unsetf(std::ios::skipws);

    // Get file size.
    wav_file.seekg(0, std::ios::end);
    size_t file_size = wav_file.tellg();
    wav_file.seekg(0, std::ios::beg);

    if (file_size <= sizeof(WavFileHeader)) {
      std::cout << "ERROR!!! Could not open wave file " << a_filename << " file size to small " << std::endl;
      exit(EXIT_READ_FILE_FAILED);
    }

    // Read header and data
    std::vector<char> buffer(file_size);
    wav_file.read(buffer.data(), file_size);
    std::memcpy(&header, buffer.data(), sizeof(WavFileHeader));
    wav_data = std::vector<char>(buffer.begin() + sizeof(WavFileHeader), buffer.end());
  }

  std::string HeaderToString() {
    std::string ret_string = "{\n";
    ret_string += "\"riff\": " + std::string(header.riff, 4) + ",\n";
    ret_string += "\"chunk_size\": " + std::to_string(header.chunk_size) + ",\n";
    ret_string += "\"wave\": " + std::string(header.wave, 4) + ",\n";
    ret_string += "\"format\": " + std::string(header.format, 4) + ",\n";
    ret_string += "\"sub_chunk_1_size\": " + std::to_string(header.sub_chunk_1_size) + ",\n";
    ret_string += "\"audio_format\": " + std::to_string(header.audio_format) + ",\n";
    ret_string += "\"num_of_channels\": " + std::to_string(header.num_of_channels) + ",\n";
    ret_string += "\"sample_rate\": " + std::to_string(header.sample_rate) + ",\n";
    ret_string += "\"bytes_per_second\": " + std::to_string(header.bytes_per_second) + ",\n";
    ret_string += "\"block_allign\": " + std::to_string(header.block_allign) + ",\n";
    ret_string += "\"bit_depth\": " + std::to_string(header.bit_depth) + ",\n";
    ret_string += "\"sub_chunk_2_id\": " + std::string(header.sub_chunk_2_id, 4) + ",\n";
    ret_string += "\"sub_chunk_2_size\": " + std::to_string(header.sub_chunk_2_size) + ",\n";
    ret_string += "\"data_size\": " + std::to_string(wav_data.size()) + "\n";
    ret_string += "}\n";
    return ret_string;
  }
  std::vector<int16_t> ToMono16BitWave() {
    // Mono.
    if (header.num_of_channels == 1) {
      // 16 bit.
      if (header.bit_depth == 16) {
        std::vector<int16_t> out_vector(static_cast<int>(wav_data.size()) / 2);
        memcpy(reinterpret_cast<char*>(out_vector.data()), wav_data.data(), static_cast<int>(wav_data.size()));
        return out_vector;
      }
      // TODO(Brandon): Handle other Bit depth.
    }

    // TODO(Brandon): Handle other formats.
    return std::vector<int16_t>(wav_data.size());
  }

  // TODO(Brandon) for future.
  std::vector<int32_t> ToMono32BitWave();
  std::vector<float> ToMonoFloatWave();

 private:
  std::string filename;
  WavFileHeader header;
  std::vector<char> wav_data;
};
}  // namespace wave
}  // namespace filereader

#endif  // INCLUDE_FILEREADER_H_
