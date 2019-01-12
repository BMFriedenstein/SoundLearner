/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * wave.h
 *  Created on: 05 Jan 2019
 *      Author: Brandon
 */

#ifndef SRC_WAVE_WAVE_H_
#define SRC_WAVE_WAVE_H_

#include <vector>
#include <memory>
#include <string>
#include "include/common.h"

class WaveReaderC {
 public:
  explicit WaveReaderC(const std::string & a_filename);
  std::string HeaderToString();
  std::vector<int16_t> ToMono16BitWave();

  // TODO(Brandon) for future.
  std::vector<int32_t> ToMono32BitWave();
  std::vector<float> ToMonoFloatWave();

 private:
  std::string filename;
  wav_header header;
  std::vector<char> wav_data;
};

class MonoWaveWriterC {
 public:
  explicit MonoWaveWriterC(const std::vector<int16_t>& a_data);
  void Write(const std::string& a_file_name);

 private:
  wav_header header;
  std::vector<char> wav_data;
};

#endif   // SRC_WAVE_WAVE_H_
