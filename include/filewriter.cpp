/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "include/filewriter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace filewriter {
namespace detail {

void EnsureOpen(const std::ofstream &stream, const std::string &filename) {
  if (!stream) {
    throw std::runtime_error("Unable to open output file: " + filename);
  }
}

uint8_t ToByte(double value) {
  return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
}

RGBA MakeRgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  RGBA rgba{};
  rgba.rgba_st.R = r;
  rgba.rgba_st.G = g;
  rgba.rgba_st.B = b;
  rgba.rgba_st.A = a;
  return rgba;
}

} // namespace detail

namespace text {

void CreateEmptyFile(const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  detail::EnsureOpen(file, filename);
}

void WriteLine(const std::string &filename, const std::string &line) {
  std::ofstream out_stream(filename, std::ios::out | std::ios::app);
  detail::EnsureOpen(out_stream, filename);
  out_stream << line << '\n';
}

void WriteFile(const std::string &filename, const std::string &content) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  detail::EnsureOpen(file, filename);
  file << content << '\n';
}

} // namespace text

namespace wave {

MonoWriter::MonoWriter(const std::vector<int16_t> &data) : wav_data(data) {}

void MonoWriter::Write(const std::string &file_name) {
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

} // namespace wave
} // namespace filewriter
