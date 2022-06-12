/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * string_oscillator.cpp
 *  Created on: 03 Jan 2019
 *    Author: Brandon
 */

#include "instrument/string_oscillator.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace instrument {
namespace oscillator {
StringOccilator::StringOccilator(double initial_phase, double frequency_factor, double amplitude_factor, double amplitude_decay,
                                 double amplitude_attack, double frequency_decay, bool is_coupled)
    : phase_factor(std::clamp(initial_phase, 0.0, 1.0)), start_frequency_factor(std::clamp(frequency_factor, 0.0, 1.0)),
      start_amplitude_factor(std::clamp(amplitude_factor, 0.0, 1.0)), amplitude_attack_factor(std::clamp(amplitude_attack, 0.0, 1.0)),
      amplitude_decay_factor(std::clamp(amplitude_decay, 0.0, 1.0)), frequency_decay_factor(std::clamp(frequency_decay, 0.0, 1.0)),
      base_frequency_coupled(is_coupled), rand_eng(std::mt19937(std::random_device{}())) {}

/*
 * Parse information required to generate a signal.
 *
 * @parameters: frequency (The base note), velocity( how hard of note was pressed  form 0-1)
 * @returns: void
 */
void StringOccilator::PrimeString(double freq, double velocity) {
  constexpr double amp_decay_range = (k_min_amp_decay_rate - k_max_amp_decay_rate);
  constexpr double amp_attack_range = (k_max_amp_attack_rate - k_min_amp_attack_rate);
  constexpr double freq_decay_range = (k_min_freq_decay_rate - k_max_freq_decay_rate);
  constexpr double amp_range = (k_max_amp_cutoff - k_min_amp_cutoff);
  const double amplitude_decay(k_max_amp_decay_rate + amp_decay_range * amplitude_decay_factor);
  const double frequency_decay(k_max_freq_decay_rate + freq_decay_range * frequency_decay_factor);
  const double amplitude_factor(k_min_amp_cutoff + amp_range * start_amplitude_factor);
  const double amplitude_attack(k_min_amp_attack_rate + amp_attack_range * amplitude_attack_factor);
  const double max_freq_factor(base_frequency_coupled ? k_max_coupled_freq_factor : k_max_uncoupled_freq_factor);
  const double frequency_factor(k_min_freq_factor + (max_freq_factor - k_min_freq_factor) * start_frequency_factor);
  amplitude_state = 0.0;
  sample_pos = 0U;
  in_amplitude_decay = false;
  max_amplitude = velocity * amplitude_factor;
  base_frequency = freq;
  frequency_state = base_frequency * frequency_factor;
  amplitude_attack_delta = amplitude_attack * max_amplitude;
  amplitude_decay_rate = amplitude_decay;
  frequency_decay_rate = frequency_decay;
  std::cout << "V" << velocity << " A" << amplitude_factor << " M " << max_amplitude << "\n";
}

/*
 * Generate the value of the next sample of the signal.
 *
 * @returns: next value of the signal float
 */
double StringOccilator::NextSample() {
  // Calculate amplitude/frequency state
  if (!in_amplitude_decay) {
    amplitude_state += amplitude_attack_delta;
  } else {
    amplitude_state *= amplitude_decay_rate;
    frequency_state *= frequency_decay_rate;
  }
  if (amplitude_state >= max_amplitude) {
    in_amplitude_decay = true;
    amplitude_state = max_amplitude;
  }

  double sample_val{0.0};
  sample_pos++;
  if (amplitude_state > k_min_amp_cutoff) {
    sample_val = SineWave();
  }
  return sample_val;
}

/*
 * Create a JSON representation of the SoundString.
 *
 * @parameters: none
 * @returns: json string
 */
std::string StringOccilator::ToJson() {
  const std::string json_str = "{\n"
                               "\"start_phase\":" +
                               std::to_string(phase_factor) +
                               ",\n"
                               "\"start_frequency_factor\":" +
                               std::to_string(start_frequency_factor) +
                               ",\n"
                               "\"start_amplitude_factor\":" +
                               std::to_string(start_amplitude_factor) +
                               ",\n"
                               "\"amp_decay_rate\":" +
                               std::to_string(amplitude_decay_factor) +
                               ",\n"
                               "\"amp_attack_delta\":" +
                               std::to_string(amplitude_attack_factor) +
                               ",\n"
                               "\"freq_decay_rate\":" +
                               std::to_string(frequency_decay_factor) +
                               ",\n"
                               "\"base_frequency_coupled\":" +
                               std::to_string(base_frequency_coupled) +
                               "\n"
                               "}";
  return json_str;
}
void StringOccilator::AmendGain(double factor) { start_amplitude_factor = std::clamp<double>(start_amplitude_factor * factor, 0.0, 1.0); }

std::string StringOccilator::ToCsv() {
  const std::string csv_str = std::to_string(start_amplitude_factor) + "," + std::to_string(start_frequency_factor) + "," +
                              std::to_string(phase_factor) + "," + std::to_string(amplitude_decay_factor) + "," +
                              std::to_string(amplitude_attack_factor) + "," + std::to_string(frequency_decay_factor) + "," +
                              std::to_string(base_frequency_coupled);
  return csv_str;
}

/*
 * Returns a mutated version of the string, each parameter of the string
 * sound only has a 50% likelihood of being mutated.
 * @parameters: severity(determines the severity of the mutation)
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOccilator> StringOccilator::TuneString(uint8_t severity) {
  const double sev_factor = static_cast<double>(severity) / 255.0;
  std::uniform_real_distribution<> real_distr(-sev_factor, sev_factor);
  const double phase = phase_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const double start_frequency = start_frequency_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const double amplitude = start_amplitude_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const double amplitude_decay = amplitude_decay_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const double amplitude_attack = amplitude_attack_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const double frequency_decay = frequency_decay_factor + ((real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0);
  const bool is_coupled = (real_distr(rand_eng) < 0.95);

  return std::make_unique<StringOccilator>(phase, start_frequency, amplitude, amplitude_decay, amplitude_attack, frequency_decay, is_coupled);
}

static std::mt19937 stat_rand_eng = std::mt19937(std::random_device{}());

/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOccilator> StringOccilator::CreateUntunedString(bool is_coupled) {
  std::uniform_real_distribution<> real_distr(0, 1); // define the range.

  const double phase = real_distr(stat_rand_eng);            // Maps to 0 to TAU
  const double freq_factor = real_distr(stat_rand_eng);      // Maps to 0 to max_uncoupled_frequency_factor  or max max_coupled_frequency_factor
  const double amplitude_factor = real_distr(stat_rand_eng); // Maps to 0 to 1
  const double amplitude_decay = real_distr(stat_rand_eng);  // Maps to min_amplitude_decay_factor to 1;
  const double amplitude_attack = real_distr(stat_rand_eng); // Maps to 0 to max Attack rate;
  const double frequency_decay = real_distr(stat_rand_eng);  // Maps to min_amplitude_decay_factor to 1;

  return std::make_unique<StringOccilator>(phase, freq_factor, amplitude_factor, amplitude_decay, amplitude_attack, frequency_decay, is_coupled);
}

/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOccilator> StringOccilator::CreateStringFromCsv(const std::string &csv_string) {
  std::stringstream string_stream(csv_string);
  std::vector<std::string> result;
  while (string_stream.good()) {
    std::string substr;
    getline(string_stream, substr, ',');
    result.push_back(std::move(substr));
  }

  std::uniform_real_distribution<> real_distr(0, 1); // define the range.
  const double amplitude_factor = result.size() > 0 ? std::stod(result[0]) : real_distr(stat_rand_eng) / 8;
  const double freq_factor = result.size() > 1 ? std::stod(result[1]) : real_distr(stat_rand_eng);
  const double phase = result.size() > 2 ? std::stod(result[2]) : real_distr(stat_rand_eng);
  const double amplitude_decay = result.size() > 4 ? std::stod(result[4]) : real_distr(stat_rand_eng);
  const double amplitude_attack = result.size() > 5 ? std::stod(result[5]) : real_distr(stat_rand_eng);
  const double frequency_decay = result.size() > 6 ? std::stod(result[6]) : real_distr(stat_rand_eng);
  const bool is_coupled = result.size() > 7 ? std::stod(result[7]) > 0.5 : true;

  return std::make_unique<StringOccilator>(phase, freq_factor, amplitude_factor, amplitude_decay, amplitude_attack, frequency_decay, is_coupled);
}
} // namespace oscillator
} // namespace instrument
