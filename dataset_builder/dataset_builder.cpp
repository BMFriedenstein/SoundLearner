/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * main.cpp
 *  Created on: 02 Jan 2019
 *      Author: Brandon
 */
#include "dataset_builder/dataset_builder.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "include/audio_features.h"
#include "include/common.h"
#include "include/filewriter.h"
#include "instrument/instrument_model.h"
#include "instrument/string_oscillator.h"

static inline void AppUsage() {
  std::cerr << "Usage: \n"
            << "-h --help\n"
            << "-n --dataset-size <number of samples to generate>\n"
            << "-c --uncoupled-oscilators <0>\n"
            << "-s --instrument-size <50>\n"
            << "-t --sample-time <5>\n"
            << "-r --resolution <512>\n"
            << "--write-ppm-preview\n"
            << "-d --data_save <'data'> (save data)"
            << "-p --startpoint <0> (save data)" << std::endl;
}

static bool ParseSize(std::string_view source, std::size_t &out) {
  const auto result = std::from_chars(source.data(), source.data() + source.size(), out);
  return result.ec == std::errc{} && result.ptr == source.data() + source.size();
}

static filewriter::image::Image ToImage(const fft::spectrogram::SpectrogramBuffer &buffer, std::size_t resolution) {
  filewriter::image::Image image(resolution, resolution);
  for (std::size_t row = 0; row < resolution; ++row) {
    for (std::size_t col = 0; col < resolution; ++col) {
      image.At(col, row) = fft::spectrogram::BufferAt(buffer, resolution, col, row);
    }
  }
  return image;
}

static filewriter::image::Image ToImage(const audio::features::FeatureTensor &features, audio::features::FeatureChannel channel) {
  filewriter::image::Image image(features.time_frames, features.frequency_bins);
  const auto channel_index = static_cast<std::size_t>(channel);
  for (std::size_t frequency_bin = 0; frequency_bin < features.frequency_bins; ++frequency_bin) {
    for (std::size_t time_frame = 0; time_frame < features.time_frames; ++time_frame) {
      image.At(time_frame, frequency_bin) = features.At(channel_index, frequency_bin, time_frame);
    }
  }
  return image;
}

static std::string DatasetMetadataJson(const std::string &sample_id, const std::string &wav_path, const std::string &feature_path,
                                       const std::string &oscillator_path, double freq, double velocity, std::size_t sample_count,
                                       std::size_t resolution, const audio::features::FeatureTensor &features, bool write_ppm_previews) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"id\": \"" << sample_id << "\",\n";
  out << "  \"audio\": {\n";
  out << "    \"path\": \"" << wav_path << "\",\n";
  out << "    \"sample_rate\": " << SAMPLE_RATE << ",\n";
  out << "    \"sample_count\": " << sample_count << "\n";
  out << "  },\n";
  out << "  \"analysis\": {\n";
  out << "    \"feature_path\": \"" << feature_path << "\",\n";
  out << "    \"format\": \"SLFT.float32.v1\",\n";
  out << "    \"frequency_scale\": \"log_fft_bin\",\n";
  out << "    \"frequency_bins\": " << features.frequency_bins << ",\n";
  out << "    \"time_frames\": " << features.time_frames << ",\n";
  out << "    \"preview_resolution\": " << resolution << ",\n";
  out << "    \"channels\": [\"log_frequency_magnitude\", \"temporal_delta\", \"onset_emphasis\"]\n";
  out << "  },\n";
  out << "  \"target\": {\n";
  out << "    \"note_frequency\": " << freq << ",\n";
  out << "    \"velocity\": " << velocity << ",\n";
  out << "    \"oscillator_csv_path\": \"" << oscillator_path << "\"\n";
  out << "  },\n";
  out << "  \"previews\": {\n";
  out << "    \"spectrogram_rgb\": \"preview/" << sample_id << "_rgb.bmp\",\n";
  out << "    \"log_frequency_rgb\": \"preview/" << sample_id << "_logfreq_rgb.bmp\"";
  if (write_ppm_previews) {
    out << ",\n";
    out << "    \"spectrogram_ppm\": \"preview/" << sample_id << ".ppm\",\n";
    out << "    \"log_frequency_ppm\": \"preview/" << sample_id << "_logfreq.ppm\"\n";
  } else {
    out << "\n";
  }
  out << "  }\n";
  out << "}\n";
  return out.str();
}

int main(int argc, char **argv) {
  // Application defaults.
  std::size_t coupled_oscilators = 50;
  std::size_t uncoupled_oscilators = 0;
  std::size_t dataset_size = 100;
  std::size_t sample_time = 5; // In seconds
  std::size_t starting_point = 0;
  std::size_t img_resolution = 512;
  bool write_ppm_previews = false;

  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    const std::string_view arg1 = argv[i];
    if ((arg1 == "-h") || (arg1 == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (arg1 == "--write-ppm-preview") {
      write_ppm_previews = true;
      continue;
    }
    if (((arg1 == "-n") || (arg1 == "--dataset-size") || (arg1 == "-m") || (arg1 == "--midi") || (arg1 == "-s") || (arg1 == "--instrument-size") ||
         (arg1 == "-d") || (arg1 == "--data_save") || (arg1 == "-c") || (arg1 == "--uncoupled-oscilators") || (arg1 == "-p") || (arg1 == "-r") ||
         (arg1 == "--resolution") ||
         (arg1 == "--startpoint")) &&
        (i + 1 < argc)) {
      const std::string_view arg2 = argv[++i];
      std::cout << arg1 << " " << arg2 << std::endl;
      if ((arg1 == "-n") || (arg1 == "--dataset-size")) {
        ParseSize(arg2, dataset_size);
      } else if ((arg1 == "-c") || (arg1 == "--uncoupled-oscilators")) {
        ParseSize(arg2, uncoupled_oscilators);
      } else if ((arg1 == "-s") || (arg1 == "--instrument-size")) {
        ParseSize(arg2, coupled_oscilators);
      } else if ((arg1 == "-t") || (arg1 == "--sample-time")) {
        ParseSize(arg2, sample_time);
      } else if ((arg1 == "-p") || (arg1 == "--startpoint")) {
        ParseSize(arg2, starting_point);
      } else if ((arg1 == "-r") || (arg1 == "--resolution")) {
        ParseSize(arg2, img_resolution);
      } else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  std::cout << "Building dataset...";
  std::cout << uncoupled_oscilators << std::endl;
  if (img_resolution == 0) {
    std::cerr << "Resolution must be greater than zero." << std::endl;
    return EXIT_BAD_ARGS;
  }

  const double note_played_freq = 1000.0;
  const double velocity = 1.0 / static_cast<double>(coupled_oscilators + uncoupled_oscilators);
  auto builder = DataBuilder(sample_time, coupled_oscilators, uncoupled_oscilators, starting_point, img_resolution, write_ppm_previews);
  for (std::size_t i = 0; i < dataset_size; ++i) {
    std::cout << "IDX: " << i << "...\n";
    builder.DataBuildJob(velocity, note_played_freq, i);
    std::cout << "\n";
  }

  return EXIT_NORMAL;
}

void DataBuilder::DataBuildJob(double velocity, double freq, std::size_t index) {
  instrument::InstrumentModel rand_instrument(coupled_oscilators, uncoupled_oscilators, std::to_string(index));
  bool sample_has_distorted = false;
  std::vector<double> sample_double = rand_instrument.GenerateSignal(velocity, freq, num_samples);
  std::vector<int16_t> sample = rand_instrument.GenerateIntSignal(velocity, freq, num_samples, sample_has_distorted, false);
  std::filesystem::create_directories("features");
  std::filesystem::create_directories("metadata");
  std::filesystem::create_directories("preview");

  // Write out the sample to a mono .wav file
  std::cout << "writing wav\n";
  const auto sample_id = std::string(data_output) + std::to_string(starting_index + index);
  const auto wav_path = sample_id + ".wav";
  const auto oscillator_path = sample_id + ".data";
  const auto feature_path = "features/" + sample_id + ".slft";
  const auto metadata_path = "metadata/" + sample_id + ".json";
  filewriter::wave::MonoWriter wave_writer(sample);
  wave_writer.Write(wav_path);

  std::cout << "writing features\n";
  const auto features = audio::features::ExtractLogFrequencyFeatures(sample_double, img_resolution, 0, num_samples);
  audio::features::WriteFeatureTensor(features, feature_path);

  // Write out the spectrogram to a image file
  std::cout << "writing previews\n";
  const auto spectogram = ToImage(fft::spectrogram::CreateSpectrogram(sample_double, img_resolution), img_resolution);
  filewriter::bmp::Write(spectogram, "preview/" + sample_id + "_rgb.bmp", ColorScaleType::RGB);

  const auto log_frequency_preview = ToImage(features, audio::features::FeatureChannel::LogFrequencyMagnitude);
  filewriter::bmp::Write(log_frequency_preview, "preview/" + sample_id + "_logfreq_rgb.bmp", ColorScaleType::RGB);

  if (write_ppm_previews) {
    filewriter::ppm::Write(spectogram, "preview/" + sample_id + ".ppm", ColorScaleType::RGB);
    filewriter::ppm::Write(log_frequency_preview, "preview/" + sample_id + "_logfreq.ppm", ColorScaleType::RGB);
  }

  // Write out meta and data files
  std::cout << "writing data\n";
  std::string instrument_data = rand_instrument.ToCsv(instrument::SortType::frequency);
  std::string instrument_meta = std::to_string(freq) + "\n";
  instrument_meta += std::to_string(velocity) + "\n";
  instrument_meta += std::to_string(img_resolution) + "\n";
  filewriter::text::WriteFile(sample_id + ".meta", instrument_meta);
  filewriter::text::WriteFile(oscillator_path, instrument_data);
  filewriter::text::WriteFile(metadata_path,
                              DatasetMetadataJson(sample_id, wav_path, feature_path, oscillator_path, freq, velocity, num_samples, img_resolution, features, write_ppm_previews));
  std::cout << "done\n";
}
