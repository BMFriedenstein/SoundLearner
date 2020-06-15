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

#include <cstdio>
#include <iostream>
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

int main(int argc, char** argv) {
  // Application defaults.
  uint16_t coupled_oscilators = 50;
  uint16_t uncoupled_oscilators = 0;
  uint16_t dataset_size = 100;
  uint32_t sample_time = 5;  // In seconds
  uint32_t starting_point = 0;

  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    const std::string arg1 = argv[i];
    if ((arg1 == "-h") || (arg1 == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (((arg1 == "-n") || (arg1 == "--dataset-size") || (arg1 == "-m") || (arg1 == "--midi") || (arg1 == "-s") ||
         (arg1 == "--instrument-size") || (arg1 == "-d") || (arg1 == "--data_save") || (arg1 == "-c") ||
         (arg1 == "--uncoupled-oscilators") || (arg1 == "-p") || (arg1 == "--startpoint")) &&
        (i + 1 < argc)) {
      const std::string arg2 = argv[++i];
      std::cout << arg1 << " " << arg2 << std::endl;
      if ((arg1 == "-n") || (arg1 == "--dataset_size")) {
        dataset_size = std::stoul(arg2);
      } else if ((arg1 == "-c") || (arg1 == "--uncoupled-oscilators")) {
        uncoupled_oscilators = std::stoul(arg2);
      } else if ((arg1 == "-s") || (arg1 == "--instrument-size")) {
        coupled_oscilators = std::stoul(arg2);
      } else if ((arg1 == "-t") || (arg1 == "--sample-time")) {
        sample_time = std::stoul(arg2);
      } else if ((arg1 == "-p") || (arg1 == "--startpoint")) {
        starting_point = std::stoul(arg2);
      } else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  std::cout << "Building dataset...";
  std::cout << uncoupled_oscilators << std::endl;
  const double note_played_freq = 1000;  //  C3 and C5
  const double velocity = 0.1;           // TODO for now+ 0.25*real_distr(eng)
  auto builder = DataBuilder(sample_time, coupled_oscilators, uncoupled_oscilators, starting_point);
  for (uint32_t i = 0; i < dataset_size;) {
    builder.DataBuildJob(note_played_freq, velocity, i);
  }

  return EXIT_NORMAL;
}

void DataBuilder::DataBuildJob(const double& velocity, const double& freq, uint32_t& index) {
  std::uniform_real_distribution<float> bool_distr;
  std::vector<bool> sustain(num_samples);
  std::string sustain_str = "";
  for (std::size_t i{0}; i < sustain.size(); i++) {
    sustain[i] = i && sustain[i] ? bool_distr(rand_eng) > 0.2 : bool_distr(rand_eng) > 0.8;
    sustain_str += std::to_string(sustain[i]) + (i < (sustain.size() - 1) ? "," : "");
  }
  instrument::InstrumentModelC rand_instrument(coupled_oscilators, uncoupled_oscilators, std::to_string(index));
  bool sample_has_distorted = false;
  std::vector<int16_t> sample_a =
      rand_instrument.GenerateIntSignal(velocity, freq, num_samples, sustain, sample_has_distorted);

  // Skip generated samples that distort.
  if (sample_has_distorted) {
    std::cout << "clipped\n" << std::endl;
  }

  // Write out the sample to a mono .wav file
  const auto file_name = std::string(data_output) + std::to_string(starting_index + index);
  filewriter::wave::MonoWaveWriterC wave_writer(sample_a);
  wave_writer.Write(file_name + ".wav");

  // Write out the spectrogram to a monochrome .bmp file
  const auto spectogram = fft::spectrogram::CreateSpectrogram<int16_t, uint32_t, img_resolution>(
      sample_a, fft_spectogram_min, fft_spectogram_max);
  filewriter::bmp::BMPWriterC<img_resolution, img_resolution> bmp_writer(spectogram);
  bmp_writer.Write(file_name + ".bmp");

  // Write out meta and data files
  std::string instrument_data = rand_instrument.ToCsv(instrument::frequency);
  std::string instrument_meta = std::to_string(freq) + "\n";
  instrument_meta += std::to_string(velocity) + "\n";
  instrument_meta += sustain_str;
  filewriter::text::WriteFile(file_name + ".meta", instrument_meta);
  filewriter::text::WriteFile(file_name + ".data", instrument_data);
  index++;
}