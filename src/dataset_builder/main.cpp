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

#include <cstdio>

#include <random>
#include <iostream>
#include <string>
#include <vector>

#include "logger/logger.h"
#include "include/common.h"
#include "instrument/string_oscillator.h"
#include "instrument/instrument_model.h"
#include "wave/wave.h"

static void AppUsage() {
  std::cerr << "Usage: \n"
            << "-h --help\n"
            << "-n --dataset-size <number of samples to generate \n"
            << "-c --uncoupled-oscilators<0> <>\n"
            << "-s --instrument-size <50>\n"
            << "-t --sample-time <2>\n"
            << "-d --data_save <'data'> (save data)"
            << "-p --startpoint <0> (save data)"
            << std::endl;
}

int main(int argc, char** argv) {
  // Application defaults.
  uint16_t coupled_oscilators = 50;
  uint16_t uncoupled_oscilators = 0;
  uint16_t dataset_size = 100;
  uint32_t sample_time = 2;  // In seconds
  uint32_t starting_point = 0;
  std::string data_output = "data";

  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (((arg == "-n") || (arg == "--dataset-size") ||
        (arg == "-m") || (arg == "--midi") ||
        (arg == "-s") || (arg == "--instrument-size") ||
        (arg == "-d") || (arg == "--data_save") ||
        (arg == "-c") || (arg == "--uncoupled-oscilators") ||
        (arg == "-p") || (arg == "--startpoint")) &&
        (i + 1 < argc)) {
      std::string arg2 = argv[++i];
      std::cout << arg << " " << arg2 << std::endl;
      if ((arg == "-n") || (arg == "--dataset_size")) {
        dataset_size = (uint16_t) std::stoul(arg2);
      }
      else if ((arg == "-c") || (arg == "--uncoupled-oscilators")) {
        uncoupled_oscilators = (uint16_t)std::stoul(arg2);
      }
      else if ((arg == "-s") || (arg == "--instrument-size")) {
        coupled_oscilators = (uint16_t)std::stoul(arg2);
      }
      else if ((arg == "-t") || (arg == "--sample-time")) {
        sample_time = (uint32_t)std::stoul(arg2);
      }
      else if ((arg == "-d") || (arg == "--data_save")) {
        data_output = arg2;
      }
      else if ((arg == "-p") || (arg == "--startpoint")) {
        starting_point = (uint32_t)std::stoul(arg2);
      }
      else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  std::random_device random_device;                   // Obtain a random number from hardware.
  std::mt19937 eng(random_device());                  // Seed the generator.
  std::uniform_real_distribution<> real_distr(0, 1);  // Define the range.
  uint32_t num_samples = SAMPLE_RATE * sample_time;
  std::vector<bool> sustain(num_samples, true);

  std::cout << "Building dataset...";
  std::cout << uncoupled_oscilators << std::endl;
  for (size_t i = 0; i < dataset_size;) {
    std::array<double, 2> notes_played {500 + 2000 * real_distr(eng), 500 + 2000 * real_distr(eng) };
    double velocity { real_distr(eng) };

    // Generate random sustain signal
    for (size_t j = 1; j < num_samples; j++) {
      sustain[j] = real_distr(eng) > 0.99 ? !sustain[j - 1] : sustain[j - 1];
    }

    // Generate two samples (at different frequencies for a randomized instrument
    // Then combine the samples
    instrument::InstrumentModelC rand_instrument(coupled_oscilators,
                                                 uncoupled_oscilators,
                                                 std::to_string(i));

    std::vector<int16_t> full_sample;
    full_sample.reserve(2 * num_samples);
    std::vector<int16_t> sample_a = rand_instrument.GenerateIntSignal(velocity,
                                                                      notes_played[0],
                                                                      num_samples,
                                                                      sustain);
    std::vector<int16_t> sample_b = rand_instrument.GenerateIntSignal(velocity,
                                                                      notes_played[1],
                                                                      num_samples,
                                                                      sustain);
    full_sample.insert(full_sample.end(), sample_a.begin(), sample_a.end());
    full_sample.insert(full_sample.end(), sample_b.begin(), sample_b.end());

    // Skip generated samples that distort.
    if (wave::SampleDoesClip(full_sample)) {
      std::cout << "clipped\n";
      continue;
    }
    std::cout << i << "\n";

    // Adjust the gains so that the velocities is distributed normally.
    // This is done as notes, higher velocities are more likely to cause distortion
    double redrisibuted_velocity = real_distr(eng);
    rand_instrument.AmendGain(velocity / redrisibuted_velocity);

    // Write out the sample to a mono .wav file
    wave::MonoWaveWriterC wave_writer(full_sample);
    wave_writer.Write(data_output + "/sample_" + std::to_string(starting_point+i) + ".wav");

    // Stringify the input parameters and save to .data file
    std::string sustain_str = "";
    for (size_t j = 0; j < num_samples; j++) {
      sustain_str += std::to_string(sustain[j]);
      if (j < num_samples - 1) {
        sustain_str += ",";
      }
    }

    // Write out meta and data files
    std::string instrument_data = rand_instrument.ToCsv(instrument::frequency);
    std::string instrument_meta = std::to_string(notes_played[0]) + "," +
                                  std::to_string(notes_played[1])  + "\n";
    instrument_meta += std::to_string(redrisibuted_velocity) + "\n";
    instrument_meta += sustain_str;
    logging::LogC::WriteFile(data_output + "/sample_" + std::to_string(starting_point+i) + ".meta",
                             instrument_meta);
    logging::LogC::WriteFile(data_output + "/sample_" + std::to_string(starting_point+i) + ".data",
                             instrument_data);

    i++;
  }

  return EXIT_NORMAL;
}

