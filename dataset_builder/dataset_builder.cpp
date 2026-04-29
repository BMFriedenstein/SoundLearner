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

#include "include/common.h"
#include "include/filewriter.h"
#include "instrument/instrument_model.h"

static inline void AppUsage() {
  std::cerr << "Usage: \n"
            << "-h --help\n"
            << "-n --dataset-size <number of samples to generate>\n"
            << "-c --uncoupled-oscilators <0>\n"
            << "-s --instrument-size <50>\n"
            << "--min-instrument-size <same as --instrument-size>\n"
            << "--max-instrument-size <same as --instrument-size>\n"
            << "--min-uncoupled-oscilators <same as --uncoupled-oscilators>\n"
            << "--max-uncoupled-oscilators <same as --uncoupled-oscilators>\n"
            << "--note-frequency <1000>\n"
            << "--min-note-frequency <same as --note-frequency>\n"
            << "--max-note-frequency <same as --note-frequency>\n"
            << "--frequency-factor <0..1>\n"
            << "--min-frequency-factor <0>\n"
            << "--max-frequency-factor <1>\n"
            << "--coupled-frequency-factors <csv of 0..1 factors>\n"
            << "--require-fundamental (force one coupled oscillator to 1.0*f0)\n"
            << "-t --sample-time <5>\n"
            << "-d --data_save <'data'> (save data)"
            << "-p --startpoint <0> (save data)" << std::endl;
}

static bool ParseSize(std::string_view source, std::size_t &out) {
  const auto result = std::from_chars(source.data(), source.data() + source.size(), out);
  return result.ec == std::errc{} && result.ptr == source.data() + source.size();
}

static bool ParseDouble(std::string_view source, double &out) {
  const auto result = std::from_chars(source.data(), source.data() + source.size(), out);
  return result.ec == std::errc{} && result.ptr == source.data() + source.size();
}

static bool ParseFrequencyFactorList(std::string_view source, std::vector<double> &out) {
  std::stringstream stream{std::string(source)};
  std::string item;
  while (std::getline(stream, item, ',')) {
    double value = 0.0;
    if (!ParseDouble(item, value) || value < 0.0 || value > 1.0) {
      return false;
    }
    out.push_back(value);
  }
  return !out.empty();
}

int main(int argc, char **argv) {
  // Application defaults.
  std::size_t min_coupled_oscilators = 50;
  std::size_t max_coupled_oscilators = 50;
  std::size_t min_uncoupled_oscilators = 0;
  std::size_t max_uncoupled_oscilators = 0;
  std::size_t dataset_size = 100;
  std::size_t sample_time = 5; // In seconds
  std::size_t starting_point = 0;
  double min_note_frequency = 1000.0;
  double max_note_frequency = 1000.0;
  double min_frequency_factor = 0.0;
  double max_frequency_factor = 1.0;
  bool require_fundamental = false;
  std::vector<double> coupled_frequency_factors;

  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    const std::string_view arg1 = argv[i];
    if ((arg1 == "-h") || (arg1 == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (arg1 == "--require-fundamental") {
      require_fundamental = true;
      continue;
    }
    if (((arg1 == "-n") || (arg1 == "--dataset-size") || (arg1 == "-m") || (arg1 == "--midi") || (arg1 == "-s") || (arg1 == "--instrument-size") ||
         (arg1 == "--min-instrument-size") || (arg1 == "--max-instrument-size") || (arg1 == "-d") || (arg1 == "--data_save") ||
         (arg1 == "-c") || (arg1 == "--uncoupled-oscilators") || (arg1 == "--min-uncoupled-oscilators") || (arg1 == "--max-uncoupled-oscilators") ||
         (arg1 == "--note-frequency") || (arg1 == "--min-note-frequency") || (arg1 == "--max-note-frequency") || (arg1 == "-p") ||
         (arg1 == "--frequency-factor") || (arg1 == "--min-frequency-factor") || (arg1 == "--max-frequency-factor") ||
         (arg1 == "--coupled-frequency-factors") ||
         (arg1 == "-t") || (arg1 == "--sample-time") || (arg1 == "--startpoint")) &&
        (i + 1 < argc)) {
      const std::string_view arg2 = argv[++i];
      std::cout << arg1 << " " << arg2 << std::endl;
      if ((arg1 == "-n") || (arg1 == "--dataset-size")) {
        ParseSize(arg2, dataset_size);
      } else if ((arg1 == "-c") || (arg1 == "--uncoupled-oscilators")) {
        ParseSize(arg2, min_uncoupled_oscilators);
        max_uncoupled_oscilators = min_uncoupled_oscilators;
      } else if ((arg1 == "-s") || (arg1 == "--instrument-size")) {
        ParseSize(arg2, min_coupled_oscilators);
        max_coupled_oscilators = min_coupled_oscilators;
      } else if (arg1 == "--min-instrument-size") {
        ParseSize(arg2, min_coupled_oscilators);
      } else if (arg1 == "--max-instrument-size") {
        ParseSize(arg2, max_coupled_oscilators);
      } else if (arg1 == "--min-uncoupled-oscilators") {
        ParseSize(arg2, min_uncoupled_oscilators);
      } else if (arg1 == "--max-uncoupled-oscilators") {
        ParseSize(arg2, max_uncoupled_oscilators);
      } else if ((arg1 == "-t") || (arg1 == "--sample-time")) {
        ParseSize(arg2, sample_time);
      } else if ((arg1 == "-p") || (arg1 == "--startpoint")) {
        ParseSize(arg2, starting_point);
      } else if (arg1 == "--note-frequency") {
        ParseDouble(arg2, min_note_frequency);
        max_note_frequency = min_note_frequency;
      } else if (arg1 == "--min-note-frequency") {
        ParseDouble(arg2, min_note_frequency);
      } else if (arg1 == "--max-note-frequency") {
        ParseDouble(arg2, max_note_frequency);
      } else if (arg1 == "--frequency-factor") {
        ParseDouble(arg2, min_frequency_factor);
        max_frequency_factor = min_frequency_factor;
      } else if (arg1 == "--min-frequency-factor") {
        ParseDouble(arg2, min_frequency_factor);
      } else if (arg1 == "--max-frequency-factor") {
        ParseDouble(arg2, max_frequency_factor);
      } else if (arg1 == "--coupled-frequency-factors") {
        if (!ParseFrequencyFactorList(arg2, coupled_frequency_factors)) {
          std::cerr << "--coupled-frequency-factors must be a comma-separated list of normalized values from 0.0 to 1.0." << std::endl;
          return EXIT_BAD_ARGS;
        }
      } else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  std::cout << "Building dataset...";
  std::cout << min_uncoupled_oscilators << "-" << max_uncoupled_oscilators << std::endl;
  if (min_coupled_oscilators > max_coupled_oscilators || min_uncoupled_oscilators > max_uncoupled_oscilators) {
    std::cerr << "Minimum oscillator counts must be less than or equal to maximum counts." << std::endl;
    return EXIT_BAD_ARGS;
  }
  if (min_note_frequency <= 0.0 || max_note_frequency <= 0.0) {
    std::cerr << "Note frequencies must be positive." << std::endl;
    return EXIT_BAD_ARGS;
  }
  if (min_frequency_factor < 0.0 || min_frequency_factor > 1.0 || max_frequency_factor < 0.0 || max_frequency_factor > 1.0) {
    std::cerr << "Frequency factors must be normalized values from 0.0 to 1.0." << std::endl;
    return EXIT_BAD_ARGS;
  }

  auto builder = DataBuilder(sample_time, min_coupled_oscilators, max_coupled_oscilators, min_uncoupled_oscilators, max_uncoupled_oscilators,
                             starting_point, min_note_frequency, max_note_frequency, min_frequency_factor, max_frequency_factor,
                             require_fundamental, coupled_frequency_factors);
  for (std::size_t i = 0; i < dataset_size; ++i) {
    builder.DataBuildJob(i);
  }

  return EXIT_NORMAL;
}

void DataBuilder::DataBuildJob(std::size_t index) {
  const auto sample_index = starting_index + index;
  const auto coupled_count = std::uniform_int_distribution<std::size_t>(min_coupled_oscilators, max_coupled_oscilators)(rand_eng);
  const auto uncoupled_count = std::uniform_int_distribution<std::size_t>(min_uncoupled_oscilators, max_uncoupled_oscilators)(rand_eng);
  const double freq = std::uniform_real_distribution<double>(min_note_frequency, max_note_frequency)(rand_eng);
  const auto oscillator_count = coupled_count + uncoupled_count;
  if (oscillator_count == 0U) {
    std::cerr << "Skipping sample " << sample_index << " because oscillator count resolved to zero." << std::endl;
    return;
  }
  const double velocity = 1.0 / static_cast<double>(oscillator_count);
  instrument::InstrumentModel rand_instrument(0, 0, std::to_string(sample_index));
  instrument::oscillator::StringOccilator::SetUntunedFrequencyFactorRange(min_frequency_factor, max_frequency_factor);
  for (std::size_t i = 0; i < uncoupled_count; ++i) {
    rand_instrument.AddUntunedString(false);
  }
  if (!coupled_frequency_factors.empty()) {
    for (std::size_t i = 0; i < coupled_count; ++i) {
      if (i < coupled_frequency_factors.size()) {
        const double factor = coupled_frequency_factors[i];
        instrument::oscillator::StringOccilator::SetUntunedFrequencyFactorRange(factor, factor);
      } else {
        instrument::oscillator::StringOccilator::SetUntunedFrequencyFactorRange(min_frequency_factor, max_frequency_factor);
      }
      rand_instrument.AddUntunedString(true);
    }
  } else if (require_fundamental && coupled_count > 0U) {
    constexpr double fundamental_factor = 1.5 / 7.0;
    instrument::oscillator::StringOccilator::SetUntunedFrequencyFactorRange(fundamental_factor, fundamental_factor);
    rand_instrument.AddUntunedString(true);
    instrument::oscillator::StringOccilator::SetUntunedFrequencyFactorRange(min_frequency_factor, max_frequency_factor);
    for (std::size_t i = 1; i < coupled_count; ++i) {
      rand_instrument.AddUntunedString(true);
    }
  } else {
    for (std::size_t i = 0; i < coupled_count; ++i) {
      rand_instrument.AddUntunedString(true);
    }
  }
  bool sample_has_distorted = false;
  std::vector<int16_t> sample = rand_instrument.GenerateIntSignal(velocity, freq, num_samples, sample_has_distorted, false);

  // Write out the sample to a mono .wav file
  std::cout << "IDX: " << sample_index << "...\n";
  const auto sample_id = std::string(data_output) + std::to_string(sample_index);
  const auto wav_path = sample_id + ".wav";
  const auto oscillator_path = sample_id + ".data";
  filewriter::wave::MonoWriter wave_writer(sample);
  wave_writer.Write(wav_path);

  // Write out meta and data files
  std::string instrument_data = rand_instrument.ToCsv(instrument::SortType::frequency);
  std::string instrument_meta = std::to_string(freq) + "\n";
  instrument_meta += std::to_string(velocity) + "\n";
  instrument_meta += std::to_string(coupled_count) + "\n";
  instrument_meta += std::to_string(uncoupled_count) + "\n";
  filewriter::text::WriteFile(sample_id + ".meta", instrument_meta);
  filewriter::text::WriteFile(oscillator_path, instrument_data);
  std::cout << "done\n";
}
