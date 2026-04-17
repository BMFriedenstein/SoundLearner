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

#include <cstdint>
#include <string>
#include <vector>

#include "include/common.h"
#include "include/structures.h"

namespace filereader {

namespace bmp {
class BMPReaderC {
public:
  explicit BMPReaderC(const std::string &a_filename);
  std::string HeaderToString();

private:
  std::string filename;
  WavFileHeader header;
  std::vector<char> wav_data;
};
} // namespace bmp

namespace wave {
class WaveReaderC {
public:
  explicit WaveReaderC(const std::string &a_filename);

  std::string HeaderToString();
  std::vector<int16_t> ToMono16BitWave();

  // TODO(Brandon) for future.
  std::vector<int32_t> ToMono32BitWave();
  std::vector<float> ToMonoFloatWave();

private:
  std::string filename;
  WavFileHeader header;
  std::vector<char> wav_data;
};
} // namespace wave
} // namespace filereader

#endif // INCLUDE_FILEREADER_H_
