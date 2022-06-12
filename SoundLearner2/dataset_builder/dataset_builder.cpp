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
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "include/common.h"
#include "include/filewriter.h"
#include "instrument/instrument_model.h"
#include "instrument/string_oscillator.h"

static inline void AppUsage() {
  std::cerr << "Usage: \n"
            << "-h --help\n"
            << "-n --dataset-size <number of samples to generate \n"
            << "-c --uncoupled-oscilators<0> <>\n"
            << "-s --instrument-size <50>\n"
            << "-t --sample-time <5>\n"
            << "-d --data_save <'data'> (save data)"
            << "-p --startpoint <0> (save data)" << std::endl;
}

int main(int argc, char **argv) {
  // Application defaults.
  std::size_t coupled_oscilators = 50;
  std::size_t uncoupled_oscilators = 0;
  std::size_t dataset_size = 100;
  std::size_t sample_time = 5; // In seconds
  std::size_t starting_point = 0;

  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    const std::string_view arg1 = argv[i];
    if ((arg1 == "-h") || (arg1 == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (((arg1 == "-n") || (arg1 == "--dataset-size") || (arg1 == "-m") || (arg1 == "--midi") || (arg1 == "-s") || (arg1 == "--instrument-size") ||
         (arg1 == "-d") || (arg1 == "--data_save") || (arg1 == "-c") || (arg1 == "--uncoupled-oscilators") || (arg1 == "-p") ||
         (arg1 == "--startpoint")) &&
        (i + 1 < argc)) {
      const std::string_view arg2 = argv[++i];
      std::cout << arg1 << " " << arg2 << std::endl;
      if ((arg1 == "-n") || (arg1 == "--dataset_size")) {
        std::from_chars(arg2.data(), arg2.data() + arg2.size(), dataset_size);
      } else if ((arg1 == "-c") || (arg1 == "--uncoupled-oscilators")) {
        std::from_chars(arg2.data(), arg2.data() + arg2.size(), uncoupled_oscilators);
      } else if ((arg1 == "-s") || (arg1 == "--instrument-size")) {
        std::from_chars(arg2.data(), arg2.data() + arg2.size(), coupled_oscilators);
      } else if ((arg1 == "-t") || (arg1 == "--sample-time")) {
        std::from_chars(arg2.data(), arg2.data() + arg2.size(), sample_time);
      } else if ((arg1 == "-p") || (arg1 == "--startpoint")) {
        std::from_chars(arg2.data(), arg2.data() + arg2.size(), starting_point);
      } else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  std::cout << "Building dataset...";
  std::cout << uncoupled_oscilators << std::endl;
  const double note_played_freq = 1000.0;
  const double velocity = 1.0 / (coupled_oscilators + uncoupled_oscilators);
  auto builder = DataBuilder(sample_time, coupled_oscilators, uncoupled_oscilators, starting_point);
  for (std::size_t i = 0; i < dataset_size; ++i) {
    std::cout << "IDX: " << i << "...\n";
    builder.DataBuildJob(velocity, note_played_freq, i);
    std::cout << "\n";
  }

  return EXIT_NORMAL;
}

void DataBuilder::DataBuildJob(double velocity, double freq, std::size_t index) {
  instrument::InstrumentModel rand_instrument(coupled_oscilators, uncoupled_oscilators, std::to_string(index));
  auto [sample, sample_has_distorted] = rand_instrument.GenerateIntSignal(velocity, freq, num_samples, false);

  // Skip generated samples that distort.
  if (sample_has_distorted) {
    std::cout << "!clipped\n";
  }

  uint32_t abs_sum;
  std::for_each(sample.begin(), sample.end(), [&abs_sum](const auto i) { abs_sum += std::max(static_cast<int16_t>(i), static_cast<int16_t>(-i)); });
  std::cout << "Amp:" << static_cast<float>(abs_sum) / (static_cast<float>(num_samples) * std::numeric_limits<int16_t>::max()) << "%\n";
  std::cout << rand_instrument.ToJson();

  // Write out the sample to a mono .wav file
  std::cout << "writing wav\n";
  const auto file_name = std::string(data_output) + std::to_string(starting_index + index);
  filewriter::wave::MonoWriter wave_writer(sample);
  wave_writer.Write(file_name + ".wav");

  // Write out the spectrogram to a image file
  std::cout << "calc spectogram\n";
  const auto spectogram = fft::spectrogram::CreateSpectrogram<int16_t, uint32_t, img_resolution>(sample);
  std::cout << "writing bmp\n";
  filewriter::bmp::BMPWriter<uint32_t, img_resolution, img_resolution> bmp_writer(spectogram);
  bmp_writer.Write<ColorScaleType::RGB>(file_name + "_rgb.bmp");
  bmp_writer.Write<ColorScaleType::YUV>(file_name + "_yuv.bmp");
  bmp_writer.Write<ColorScaleType::GRAYSCALE>(file_name + "_grey.bmp");
  std::cout << "writing ppm\n";
  filewriter::ppm::PPMWriter<uint32_t, img_resolution, img_resolution, ColorScaleType::RGB> ppm_writer(std::move(spectogram));
  ppm_writer.Write(file_name + ".ppm");

  const auto mel_spectogram = fft::spectrogram::CreateMelSpectrogram<img_resolution>(sample);
  std::cout << "writing bmp\n";
  filewriter::bmp::BMPWriter<uint32_t, img_resolution, img_resolution> mbmp_writer(mel_spectogram);
  mbmp_writer.Write("mel" + file_name + ".bmp");
  std::cout << "writing ppm\n";
  filewriter::ppm::PPMWriter<uint32_t, img_resolution, img_resolution> mppm_writer(std::move(mel_spectogram));
  mppm_writer.Write("mel" + file_name + ".ppm");

  // Write out meta and data files
  std::cout << "writing data\n";
  std::string instrument_data = rand_instrument.ToCsv(instrument::SortType::frequency);
  std::string instrument_meta = std::to_string(freq) + "\n";
  instrument_meta += std::to_string(velocity) + "\n";
  filewriter::text::WriteFile(file_name + ".meta", instrument_meta);
  filewriter::text::WriteFile(file_name + ".data", instrument_data);
  std::cout << "done\n";
}
