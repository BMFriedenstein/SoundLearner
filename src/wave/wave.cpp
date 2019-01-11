/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * wave.cpp
 *  Created on: 05 Jan 2019
 *      Author: Brandon
 */

#include "wave/wave.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>

WaveReaderC::WaveReaderC(const std::string & a_filename) {
  filename = a_filename;

  // Open file
  std::ifstream wav_file(a_filename, std::ios::binary);
  if (!wav_file.is_open()) {
    std::cout << "ERROR!!! Could not open wave file " << a_filename
              << std::endl;
    exit(EXIT_READ_FILE_FAILED);
  }
  wav_file.unsetf(std::ios::skipws);

  // Get file size
  wav_file.seekg(0, std::ios::end);
  size_t file_size = wav_file.tellg();
  wav_file.seekg(0, std::ios::beg);

  if (file_size <= sizeof(wav_header)) {
    std::cout << "ERROR!!! Could not open wave file " << a_filename
              << " file size to small " << std::endl;
    exit(EXIT_READ_FILE_FAILED);
  }

  // Read header and data
  std::vector<char> buffer(file_size);
  std::cout << "DEBUG : header_size: " << sizeof(wav_header) << std::endl;
  wav_file.read(buffer.data(), file_size);
  std::memcpy(&header, buffer.data(), sizeof(wav_header));
  wav_data = std::vector<char>(buffer.begin() + sizeof(wav_header),
                               buffer.end());
}

std::string WaveReaderC::HeaderToString() {
  std::string ret_string = "{\n";
  ret_string += "\"riff\": " + std::string(header.riff, 4) + ",\n";
  ret_string += "\"chunk_size\": " + std::to_string(header.chunk_size) + ",\n";
  ret_string += "\"wave\": " + std::string(header.wave, 4) + ",\n";
  ret_string += "\"format\": " + std::string(header.format, 4) + ",\n";
  ret_string += "\"sub_chunk_1_size\": "
      + std::to_string(header.sub_chunk_1_size) + ",\n";
  ret_string += "\"audio_format\": " + std::to_string(header.audio_format)
      + ",\n";
  ret_string += "\"num_of_channels\": " + std::to_string(header.num_of_channels)
      + ",\n";
  ret_string += "\"sample_rate\": " + std::to_string(header.sample_rate)
      + ",\n";
  ret_string += "\"bytes_per_second\": "
      + std::to_string(header.bytes_per_second) + ",\n";
  ret_string += "\"block_allign\": " + std::to_string(header.block_allign)
      + ",\n";
  ret_string += "\"bit_depth\": " + std::to_string(header.bit_depth) + ",\n";
  ret_string += "\"sub_chunk_2_id\": " + std::string(header.sub_chunk_2_id, 4)
      + ",\n";
  ret_string += "\"sub_chunk_2_size\": "
      + std::to_string(header.sub_chunk_2_size) + ",\n";
  ret_string += "\"data_size\": " + std::to_string(wav_data.size()) + "\n";
  ret_string += "}\n";
  return ret_string;
}

std::vector<int16_t> WaveReaderC::ToMono16BitWave() {
  // Mono.
  if (header.num_of_channels == 1) {
    // 16 bit.
    if (header.bit_depth == 16) {
      std::vector<int16_t> out_vector(static_cast<int>(wav_data.size()) / 2);
      memcpy(reinterpret_cast<char*>(out_vector.data()), wav_data.data(),
             static_cast<int>(wav_data.size()));
      return out_vector;
    }
    // TODO(Brandon): Handle other Bit depth
  }

  // TODO(Brandon) Handle other formats
  return std::vector<int16_t>(wav_data.size());
}

MonoWaveWriterC::MonoWaveWriterC(const std::vector<int16_t>& a_data) {
  wav_data.resize(a_data.size() * 2);
  memcpy(reinterpret_cast<char*>(wav_data.data()),
         reinterpret_cast<const char*>(a_data.data()),
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

void MonoWaveWriterC::Write(const std::string& a_file_name) {
  std::vector<char> data(wav_data.size() + sizeof(wav_header));
  memcpy(data.data(), &header, sizeof(wav_header));
  memcpy(data.data() + sizeof(wav_header), wav_data.data(), wav_data.size());
  std::ofstream fout(a_file_name, std::ios::out | std::ios::binary);
  fout.write(data.data(), data.size());
  fout.close();
}
