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
#include <limits>
#include <iostream>
#include <string>
#include <random>
#include <utility>

namespace instrument {
InstrumentModelC::InstrumentModelC(const uint16_t num_strings,
                                   const std::string& instrument_name) {
  error_score_ = std::numeric_limits<double>::max();
  name_ = instrument_name;
  sound_strings_.reserve(num_strings);
  for (uint16_t i = 0; i < num_strings; i++) {
    AddUntunedString();
  }
}

/*
 * Add a pre tuned string to the instrument
 * @params: reference to a a_tuned_string (StringOscillatorC),
 * @returns: none
 */
void InstrumentModelC::AddTunedString(
    const oscillator::StringOscillatorC a_tuned_string) {
  std::unique_ptr<oscillator::StringOscillatorC> tuned_string(
      new oscillator::StringOscillatorC(a_tuned_string));
  sound_strings_.push_back(std::move(tuned_string));
}

/*
 * Add a randomly tuned string sound to the instrument
 * @params: none,
 * @returns: none
 */
void InstrumentModelC::AddUntunedString() {
  sound_strings_.push_back(
      std::move(oscillator::StringOscillatorC::CreateUntunedString()));
}

/*
 * Create a JSON representation of the instrument
 * @params: none,
 * @returns: JSON string
 */
std::string InstrumentModelC::ToJson() {
  std::string return_json = "{\n";
  return_json += "\"name\": \"" + name_ + "\",\n";
  return_json += "\"strings\": {\n";
  for (size_t j = 0; j < sound_strings_.size(); j++) {
    return_json += sound_strings_[j]->ToJson();
    if (j + 1 != sound_strings_.size()) {
      return_json += ",\n";
    }
  }
  return_json += "}\n}\n";
  return return_json;
}

/*
 * Generates a array of double sample values representing
 * the sound of the note played
 * @params: velocity(speed of note played), frequency(Which note),
 *          number of sample to generate, array of sustain values
 * @returns: vector of doubles
 */
std::vector<double> InstrumentModelC::GenerateSignal(
    const double velocity, const double frequency,
    const uint32_t num_of_samples, std::vector<bool>& sustain) {
  std::vector<double> signal(num_of_samples);

  // Check that we have a sustain value for each sample
  if (sustain.size() != num_of_samples) {
    std::cout << "Warning!!! Sustain array not equal to sample length"
              << std::endl;
    sustain.resize(num_of_samples);
  }

  // Initiate each of the strings
  for (size_t i = 0; i < sound_strings_.size(); i++) {
    sound_strings_[i]->PrimeString(frequency, velocity);
  }

  // Generate samples
  for (uint32_t i = 0; i < num_of_samples; i++) {
    double sample_val = 0;

    for (size_t j = 0; j < sound_strings_.size(); j++) {
      sample_val += sound_strings_[j]->NextSample(sustain[i]);
    }
    signal[i] = sample_val;
  }

  return signal;
}

/*
 * Generates a array of rounded integer sample values representing
 * the sound of the note played
 * @params: velocity(speed of note played), frequency(Which note),
 *          number of sample to generate, array of sustain values
 * @returns: vector of integers
 */
std::vector<int16_t> InstrumentModelC::GenerateIntSignal(
    const double velocity, const double frequency,
    const uint32_t num_of_samples, std::vector<bool>& sustain) {
  std::vector<int16_t> signal(num_of_samples);

  // Check that we have a sustain value for each sample
  if (sustain.size() != num_of_samples) {
    std::cout << "Warning!!! Sustain array not equal to sample length"
              << std::endl;
    sustain.resize(num_of_samples);
  }

  // Initiate each of the strings
  for (size_t i = 0; i < sound_strings_.size(); i++) {
    sound_strings_[i]->PrimeString(frequency, velocity);
  }

  // Generate samples
  for (uint32_t i = 0; i < num_of_samples; i++) {
    double sample_val = 0;
    for (size_t j = 0; j < sound_strings_.size(); j++) {
      sample_val += MAX_AMP * sound_strings_[j]->NextSample(sustain[i]);
    }

    // Convert to int
    if (sample_val > MAX_AMP) {
      sample_val = MAX_AMP;
    } else if (sample_val < MIN_AMP) {
      sample_val = MIN_AMP;
    }
    if (sample_val < 0) {
      signal[i] = static_cast<int>(sample_val - 0.5);
    } else {
      signal[i] = static_cast<int>(sample_val + 0.5);
    }
  }

  return signal;
}

/*
 * Create a new instrument slightly mutated from this instrument model
 * @params: amount( mutation amount)
 * @returns: unique pointer to an instrument
 */
std::unique_ptr<InstrumentModelC> InstrumentModelC::TuneInstrument(
    const uint8_t amount) {
  std::unique_ptr<InstrumentModelC> mutant_instrument;
  std::random_device random_device;      // obtain a random number from hardware
  std::mt19937 eng(random_device());                  // seed the generator
  std::uniform_real_distribution<> real_distr(0, 1);  // define the range
  bool create_new = real_distr(eng) < 0.1;

  if (create_new) {
    mutant_instrument = std::unique_ptr<InstrumentModelC>(
        new InstrumentModelC(sound_strings_.size(),
                             "new_" + std::to_string(time(nullptr))));
  } else {
    mutant_instrument = std::unique_ptr<InstrumentModelC>(
        new InstrumentModelC(0, "new_" + std::to_string(time(nullptr))));
    for (size_t j = 0; j < sound_strings_.size(); j++) {
      bool keep_tune = real_distr(eng) < 0.1;
      if (!keep_tune) {
        mutant_instrument->AddTunedString(
            *sound_strings_[j]->TuneString(amount));
      } else {
        mutant_instrument->AddTunedString(*sound_strings_[j]);
      }
    }
  }
  return mutant_instrument;
}
}  // namespace instrument
