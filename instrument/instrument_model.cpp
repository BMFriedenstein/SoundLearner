/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * instrument_model.cpp
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */

#include "instrument/instrument_model.h"

#include <cstdio>

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>

namespace instrument {

InstrumentModelC::InstrumentModelC(const std::vector<std::string>& csv_strings, const std::string& a_name)
    : error_score(std::numeric_limits<double>::max()),
      mae_score(std::numeric_limits<double>::max()),
      corr_score(std::numeric_limits<double>::max()),
      diff_score(std::numeric_limits<double>::max()),
      name(a_name) {
  sound_strings.reserve(csv_strings.size());
  std::for_each(csv_strings.begin(), csv_strings.end(), [&](const auto& str) {
    sound_strings.push_back(std::move(oscillator::StringOscillatorC::CreateStringFromCsv(str)));
  });
}

InstrumentModelC::InstrumentModelC(uint16_t num_strings, const std::string& instrument_name) {
  error_score = std::numeric_limits<double>::max();
  name = instrument_name;
  sound_strings.reserve(num_strings);
  for (uint16_t i = 0; i < num_strings; i++) {
    AddUntunedString(false);
  }
}

InstrumentModelC::InstrumentModelC(uint16_t num_coupled_strings,
                                   uint16_t num_uncoupled_strings,
                                   const std::string& instrument_name) {
  error_score = std::numeric_limits<double>::max();
  name = instrument_name;
  sound_strings.reserve(num_uncoupled_strings + num_coupled_strings);
  for (uint16_t i = 0; i < num_uncoupled_strings; i++) {
    AddUntunedString(false);
  }
  for (uint16_t i = 0; i < num_coupled_strings; i++) {
    AddUntunedString(true);
  }
}

/*
 * Add a pre-tuned string to the instrument.
 * @parameters: reference to a a_tuned_string (StringOscillatorC),
 * @returns: none
 */
void InstrumentModelC::AddTunedString(const oscillator::StringOscillatorC&& a_tuned_string) {
  auto tuned_string = std::make_unique<oscillator::StringOscillatorC>(oscillator::StringOscillatorC(a_tuned_string));
  sound_strings.push_back(std::move(tuned_string));
}

/*
 * Add a randomly tuned string sound to the instrument.
 * @parameters: none,
 * @returns: none
 */
void InstrumentModelC::AddUntunedString(bool is_coupled) {
  sound_strings.push_back(std::move(oscillator::StringOscillatorC::CreateUntunedString(is_coupled)));
}

/*
 * Create a JSON representation of the instrument.
 * @parameters: none,
 * @returns: JSON string
 */
std::string InstrumentModelC::ToJson(SortType sort_type) {
  std::string return_json = "{\n";
  if (sort_type == SortType::frequency) {
    SortStringsByFreq();
  } else if (sort_type == SortType::amplitude) {
    SortStringsByAmplitude();
  }
  return_json += "\"name\": \"" + name + "\",\n";
  return_json += "\"strings\": [\n";
  for (std::size_t j = 0; j < sound_strings.size(); j++) {
    return_json += sound_strings[j]->ToJson();
    if (j + 1 != sound_strings.size()) {
      return_json += ",\n";
    }
  }
  return_json += "]\n}\n";
  return return_json;
}

std::string InstrumentModelC::ToCsv(SortType sort_type) {
  std::string return_csv = "";
  if (sort_type == SortType::frequency) {
    SortStringsByFreq();
  } else if (sort_type == SortType::amplitude) {
    SortStringsByAmplitude();
  }
  for (std::size_t j = 0; j < sound_strings.size(); j++) {
    return_csv += sound_strings[j]->ToCsv();
    if (j + 1 != sound_strings.size()) {
      return_csv += "\n";
    }
  }

  return return_csv;
}

/*
 * Generates a array of double sample values representing
 * the sound of the note played.
 * @parameters: velocity(speed of note played), frequency(Which note),
 *          number of sample to generate.
 * @returns: vector of doubles
 */
std::vector<double> InstrumentModelC::GenerateSignal(double velocity, double frequency, uint32_t num_of_samples) {
  std::for_each(sound_strings.begin(), sound_strings.end(), [&](const auto& s) { s->PrimeString(frequency, velocity); });

  // Generate samples.
  std::vector<double> signal(num_of_samples);
  for (uint32_t i = 0; i < signal.size(); ++i) {
    double sample_val{0};
    std::for_each(sound_strings.begin(), sound_strings.end(), [&](const auto& s) { sample_val += s->NextSample(); });
    signal[i] = sample_val;
  }

  return signal;
}

/*
 * Generates a array of rounded integer sample values representing
 * the sound of the note played.
 * @parameters: velocity(speed of note played), frequency(Which note),
 *          number of sample to generate.
 * @returns: vector of integers
 */
std::vector<int16_t> InstrumentModelC::GenerateIntSignal(double velocity,
                                                         double frequency,
                                                         uint32_t num_of_samples,
                                                         bool& has_distorted_out,
                                                         bool return_on_distort) {
  has_distorted_out = false;
  std::for_each(sound_strings.begin(), sound_strings.end(), [&](const auto& s) { s->PrimeString(frequency, velocity); });

  // Generate samples.
  std::vector<int16_t> signal(num_of_samples);
  for (uint32_t i = 0; i < num_of_samples; i++) {
    double sample_val = 0;
    std::for_each(sound_strings.begin(), sound_strings.end(), [&](const auto& s) { sample_val += s->NextSample(); });

    // Convert to int32.
    if (sample_val > MAX_AMP) {
      sample_val = MAX_AMP;
      has_distorted_out = true;
    } else if (sample_val < MIN_AMP) {
      sample_val = MIN_AMP;
      has_distorted_out = true;
    }
    if (return_on_distort && has_distorted_out) {
      return signal;
    }

    if (sample_val < 0) {
      signal[i] = static_cast<int>(sample_val - 0.5);
    } else {
      signal[i] = static_cast<int>(sample_val + 0.5);
    }
  }

  return signal;
}

void InstrumentModelC::AmendGain(double factor) {
  std::for_each(sound_strings.begin(), sound_strings.end(), [&](const auto& s) { s->AmendGain(factor); });
}
static std::mt19937 stat_rand_eng = std::mt19937(std::random_device {}());
/*
 * Create a new instrument slightly mutated from this instrument model.
 * @parameters: amount(mutation amount).
 * @returns: unique pointer to an instrument
 */
std::unique_ptr<InstrumentModelC> InstrumentModelC::TuneInstrument(uint8_t amount) {
  std::uniform_real_distribution<> real_distr(0, 1);  // define the range.

  std::string new_name = "new_" + std::to_string(time(nullptr));
  if (real_distr(stat_rand_eng) < 0.1) {
    return std::make_unique<InstrumentModelC>(InstrumentModelC(sound_strings.size(), new_name));
  } else {
    auto mutant_instrument = std::make_unique<InstrumentModelC>(InstrumentModelC(0, new_name));
    for (std::size_t j = 0; j < sound_strings.size(); j++) {
      if (real_distr(stat_rand_eng) < 0.95) {
        mutant_instrument->AddTunedString(std::move(*sound_strings[j]->TuneString(amount).release()));
      } else {
        mutant_instrument->AddTunedString(std::move(*sound_strings[j]));
      }
    }
    return mutant_instrument;
  }
}
}  // namespace instrument
