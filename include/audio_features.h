/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once
#ifndef INCLUDE_AUDIO_FEATURES_H_
#define INCLUDE_AUDIO_FEATURES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace audio::features {

enum class FeatureChannel : uint32_t {
  LogFrequencyMagnitude = 0,
  TemporalDelta = 1,
  OnsetEmphasis = 2,
};

struct FeatureTensor {
  std::size_t channels = 0;
  std::size_t frequency_bins = 0;
  std::size_t time_frames = 0;
  std::vector<float> data;

  float &At(std::size_t channel, std::size_t frequency_bin, std::size_t time_frame);
  float At(std::size_t channel, std::size_t frequency_bin, std::size_t time_frame) const;
};

std::vector<double> CropOrPad(const std::vector<double> &source_signal, std::size_t start_sample, std::size_t sample_count);
FeatureTensor ExtractLogFrequencyFeatures(const std::vector<double> &source_signal, std::size_t resolution, std::size_t fft_size_multiplier = 25);
FeatureTensor ExtractLogFrequencyFeatures(const std::vector<double> &source_signal, std::size_t resolution, std::size_t start_sample, std::size_t sample_count,
                                          std::size_t fft_size_multiplier = 25);
FeatureTensor ExtractLogFrequencyFeatureGrid(const std::vector<double> &source_signal, std::size_t frequency_bins, std::size_t time_frames,
                                             std::size_t fft_size_multiplier = 25);
FeatureTensor ExtractLogFrequencyFeatureGrid(const std::vector<double> &source_signal, std::size_t frequency_bins, std::size_t time_frames, std::size_t start_sample,
                                             std::size_t sample_count, std::size_t fft_size_multiplier = 25);
void WriteFeatureTensor(const FeatureTensor &features, const std::string &file_name);

} // namespace audio::features

#endif // INCLUDE_AUDIO_FEATURES_H_
