/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "include/audio_features.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

#include "include/common.h"
#include "include/fft.h"

namespace audio::features {
namespace {

struct FeatureTensorHeader {
  std::array<char, 4> magic{'S', 'L', 'F', 'T'};
  uint32_t version = 1;
  uint32_t sample_rate = SAMPLE_RATE;
  uint32_t channels = 0;
  uint32_t frequency_bins = 0;
  uint32_t time_frames = 0;
};

float Clamp01(double value) {
  return static_cast<float>(std::clamp(value, 0.0, 1.0));
}

} // namespace

float &FeatureTensor::At(std::size_t channel, std::size_t frequency_bin, std::size_t time_frame) {
  return data[(channel * frequency_bins + frequency_bin) * time_frames + time_frame];
}

float FeatureTensor::At(std::size_t channel, std::size_t frequency_bin, std::size_t time_frame) const {
  return data[(channel * frequency_bins + frequency_bin) * time_frames + time_frame];
}

std::vector<double> CropOrPad(const std::vector<double> &source_signal, std::size_t start_sample, std::size_t sample_count) {
  std::vector<double> cropped(sample_count);
  if (start_sample >= source_signal.size()) {
    return cropped;
  }

  const auto copy_size = std::min(sample_count, source_signal.size() - start_sample);
  std::copy_n(source_signal.begin() + static_cast<std::ptrdiff_t>(start_sample), copy_size, cropped.begin());
  return cropped;
}

FeatureTensor ExtractLogFrequencyFeatures(const std::vector<double> &source_signal, std::size_t resolution) {
  constexpr std::size_t channel_count = 3;
  FeatureTensor features{
      .channels = channel_count,
      .frequency_bins = resolution,
      .time_frames = resolution,
      .data = std::vector<float>(channel_count * resolution * resolution),
  };

  const auto magnitude = fft::spectrogram::CreateMelSpectrogram(source_signal, resolution);

  for (std::size_t frequency_bin = 0; frequency_bin < resolution; ++frequency_bin) {
    for (std::size_t time_frame = 0; time_frame < resolution; ++time_frame) {
      const double current = fft::spectrogram::BufferAt(magnitude, resolution, time_frame, frequency_bin);
      const double previous = time_frame == 0 ? current : fft::spectrogram::BufferAt(magnitude, resolution, time_frame - 1, frequency_bin);
      const double delta = current - previous;

      features.At(static_cast<std::size_t>(FeatureChannel::LogFrequencyMagnitude), frequency_bin, time_frame) = Clamp01(current);
      features.At(static_cast<std::size_t>(FeatureChannel::TemporalDelta), frequency_bin, time_frame) = Clamp01((delta + 1.0) * 0.5);
      features.At(static_cast<std::size_t>(FeatureChannel::OnsetEmphasis), frequency_bin, time_frame) = Clamp01(std::max(delta, 0.0));
    }
  }

  return features;
}

FeatureTensor ExtractLogFrequencyFeatures(const std::vector<double> &source_signal, std::size_t resolution, std::size_t start_sample, std::size_t sample_count) {
  return ExtractLogFrequencyFeatures(CropOrPad(source_signal, start_sample, sample_count), resolution);
}

void WriteFeatureTensor(const FeatureTensor &features, const std::string &file_name) {
  std::ofstream out(file_name, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to open feature tensor file: " + file_name);
  }

  const FeatureTensorHeader header{
      .channels = static_cast<uint32_t>(features.channels),
      .frequency_bins = static_cast<uint32_t>(features.frequency_bins),
      .time_frames = static_cast<uint32_t>(features.time_frames),
  };

  out.write(reinterpret_cast<const char *>(&header), sizeof(header));
  out.write(reinterpret_cast<const char *>(features.data.data()), static_cast<std::streamsize>(features.data.size() * sizeof(float)));
}

} // namespace audio::features
