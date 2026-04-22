/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <charconv>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "include/audio_features.h"
#include "include/common.h"
#include "include/filereader.h"
#include "include/filewriter.h"

namespace {

void AppUsage() {
  std::cerr << "Usage:\n"
            << "-h --help\n"
            << "-i --input <input.wav>\n"
            << "-o --output <features.slft>\n"
            << "-r --resolution <512>\n"
            << "--freq-bins <512>\n"
            << "--time-frames <512>\n"
            << "--fft-size-multiplier <25>\n"
            << "-t --crop-seconds <5>\n"
            << "--crop-start-seconds <0>\n"
            << "-p --preview-prefix <preview/name>\n"
            << "--write-ppm-preview\n";
}

bool ParseSize(std::string_view source, std::size_t &out) {
  const auto result = std::from_chars(source.data(), source.data() + source.size(), out);
  return result.ec == std::errc{} && result.ptr == source.data() + source.size();
}

std::vector<double> ToNormalizedDoubleSamples(const std::vector<int16_t> &pcm) {
  std::vector<double> samples;
  samples.reserve(pcm.size());

  constexpr double max_sample = static_cast<double>(std::numeric_limits<int16_t>::max());
  for (const auto sample : pcm) {
    samples.push_back(static_cast<double>(sample) / max_sample);
  }

  return samples;
}

filewriter::image::Image FeatureChannelToImage(const audio::features::FeatureTensor &features, audio::features::FeatureChannel channel) {
  filewriter::image::Image image(features.time_frames, features.frequency_bins);
  const auto channel_index = static_cast<std::size_t>(channel);
  for (std::size_t frequency_bin = 0; frequency_bin < features.frequency_bins; ++frequency_bin) {
    for (std::size_t time_frame = 0; time_frame < features.time_frames; ++time_frame) {
      image.At(time_frame, frequency_bin) = features.At(channel_index, frequency_bin, time_frame);
    }
  }
  return image;
}

void WritePreviews(const audio::features::FeatureTensor &features, const std::string &prefix, bool write_ppm_preview) {
  const std::filesystem::path prefix_path(prefix);
  if (prefix_path.has_parent_path()) {
    std::filesystem::create_directories(prefix_path.parent_path());
  }

  const auto log_frequency = FeatureChannelToImage(features, audio::features::FeatureChannel::LogFrequencyMagnitude);

  filewriter::bmp::Write(log_frequency, prefix + "_rgb.bmp", ColorScaleType::RGB);
  if (write_ppm_preview) {
    filewriter::ppm::Write(log_frequency, prefix + ".ppm", ColorScaleType::RGB);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string input_file;
  std::string output_file;
  std::string preview_prefix;
  std::size_t resolution = 512;
  std::size_t frequency_bins = 512;
  std::size_t time_frames = 512;
  std::size_t fft_size_multiplier = 25;
  std::size_t crop_seconds = 5;
  std::size_t crop_start_seconds = 0;
  bool write_ppm_preview = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }

    if (arg == "--write-ppm-preview") {
      write_ppm_preview = true;
      continue;
    }

    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << arg << '\n';
      return EXIT_BAD_ARGS;
    }

    const std::string_view value = argv[++i];
    if ((arg == "-i") || (arg == "--input")) {
      input_file = std::string(value);
    } else if ((arg == "-o") || (arg == "--output")) {
      output_file = std::string(value);
    } else if ((arg == "-r") || (arg == "--resolution")) {
      if (!ParseSize(value, resolution)) {
        std::cerr << "Invalid resolution: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
      frequency_bins = resolution;
      time_frames = resolution;
    } else if (arg == "--freq-bins") {
      if (!ParseSize(value, frequency_bins)) {
        std::cerr << "Invalid frequency bins: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
    } else if (arg == "--time-frames") {
      if (!ParseSize(value, time_frames)) {
        std::cerr << "Invalid time frames: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
    } else if (arg == "--fft-size-multiplier") {
      if (!ParseSize(value, fft_size_multiplier)) {
        std::cerr << "Invalid FFT size multiplier: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
    } else if ((arg == "-t") || (arg == "--crop-seconds")) {
      if (!ParseSize(value, crop_seconds)) {
        std::cerr << "Invalid crop seconds: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
    } else if (arg == "--crop-start-seconds") {
      if (!ParseSize(value, crop_start_seconds)) {
        std::cerr << "Invalid crop start seconds: " << value << '\n';
        return EXIT_BAD_ARGS;
      }
    } else if ((arg == "-p") || (arg == "--preview-prefix")) {
      preview_prefix = std::string(value);
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      return EXIT_BAD_ARGS;
    }
  }

  if (input_file.empty() || output_file.empty() || frequency_bins == 0 || time_frames == 0 || crop_seconds == 0 || fft_size_multiplier == 0) {
    AppUsage();
    return EXIT_BAD_ARGS;
  }

  const std::filesystem::path output_path(output_file);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  filereader::wave::WaveReaderC reader(input_file);
  const auto pcm = reader.ToMono16BitWave();
  const auto samples = ToNormalizedDoubleSamples(pcm);
  const auto start_sample = crop_start_seconds * SAMPLE_RATE;
  const auto sample_count = crop_seconds * SAMPLE_RATE;
  const auto features = audio::features::ExtractLogFrequencyFeatureGrid(samples, frequency_bins, time_frames, start_sample, sample_count, fft_size_multiplier);
  audio::features::WriteFeatureTensor(features, output_file);

  if (!preview_prefix.empty()) {
    WritePreviews(features, preview_prefix, write_ppm_preview);
  }

  std::cout << "Wrote " << output_file << " (" << features.channels << " x " << features.frequency_bins << " x " << features.time_frames << ", crop " << crop_seconds
            << "s from " << crop_start_seconds << "s)\n";
  return EXIT_NORMAL;
}
