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
#include <vector>
namespace instrument {
namespace oscillator {
StringOscillatorC::StringOscillatorC(const double& initial_phase,
                                     const double& frequency_factor,
                                     const double& amplitude_factor,
                                     const double& non_sustain_factor,
                                     const double& amplitude_decay,
                                     const double& amplitude_attack,
                                     const double& frequency_decay,
                                     bool is_coupled)
    : phase_factor(std::clamp(initial_phase, 0.0, 1.0)),
      start_frequency_factor(std::clamp(frequency_factor, 0.0, 1.0)),
      start_amplitude_factor(std::clamp(amplitude_factor, 0.0, 1.0)),
      non_sustain_factor(std::clamp(non_sustain_factor, 0.0, 1.0)),
      amplitude_attack_factor(std::clamp(amplitude_attack, 0.0, 1.0)),
      amplitude_decay_factor(std::clamp(amplitude_decay, 0.0, 1.0)),
      frequency_decay_factor(std::clamp(frequency_decay, 0.0, 1.0)),
      base_frequency_coupled(is_coupled),
      rand_eng(std::mt19937(std::random_device {}())) {}

/*
 * Parse information required to generate a signal.
 *
 * @parameters: frequency (The base note), velocity( how hard of note was pressed  form 0-1)
 * @returns: void
 */
void StringOscillatorC::PrimeString(const double& freq, const double& velocity) {
  const double amplitude_decay(k_max_amp_decay_rate +
                               (k_min_amp_decay_rate - k_max_amp_decay_rate) * amplitude_decay_factor);
  const double frequency_decay(k_max_freq_decay_rate + (k_min_freq_decay_rate - k_max_freq_decay_rate) * frequency_decay_factor);
  const double sustain_decay(k_max_freq_decay_rate + (k_min_freq_decay_rate - k_max_freq_decay_rate) * non_sustain_factor);
  const double amplitude_factor(k_min_amp_cutoff + (k_max_amp_cutoff - k_min_amp_cutoff) * start_amplitude_factor);
  const double max_freq_factor(base_frequency_coupled ? k_max_coupled_freq_factor : k_max_uncoupled_freq_factor);
  const double frequency_factor(k_min_freq_factor + (max_freq_factor - k_min_freq_factor) * start_frequency_factor);
  const double amplitude_attack(k_min_amp_attack_rate + (k_max_amp_attack_rate - k_min_amp_attack_rate) * amplitude_attack_factor);

  amplitude_state = 0;
  sample_pos = 0;
  in_amplitude_decay = false;
  max_amplitude = velocity * amplitude_factor;
  base_frequency = freq;
  frequency_state = base_frequency * frequency_factor;
  amplitude_attack_delta = amplitude_attack * max_amplitude;
  normal_amplitude_decay_rate = amplitude_decay;
  sutain_amplitude_decay_rate = sustain_decay;
  frequency_decay_rate = frequency_decay;
}

/*
 * Generate the value of the next sample of the signal.
 *
 * @parameters: sustain (Is the note still being pressed)
 * @returns: next value of the signal float
 */
double StringOscillatorC::NextSample(bool sustain) {
  // Calculate amplitude/frequency state
  if (!in_amplitude_decay) {
    amplitude_state += amplitude_attack_delta;
  } else {
    amplitude_state *= sustain ? sutain_amplitude_decay_rate : normal_amplitude_decay_rate;
    frequency_state *= frequency_decay_rate;
  }
  if (amplitude_state >= max_amplitude) {
    in_amplitude_decay = true;
    amplitude_state = max_amplitude;
  }

  if (frequency_state > k_max_uncoupled_freq_factor || amplitude_state < k_min_amp_cutoff) {
    sample_pos += k_sample_increment;
    return 0.0;
  } else {
    // generate sample.
    double sample_val = SineWave();
    // Apply some filtering
    if (frequency_state > 15000) {
      sample_val = sample_val * (10 - 0.0005 * frequency_state);
    }
    sample_pos += k_sample_increment;
    return sample_val;
  }
}

/*
 * Create a JSON representation of the SoundString.
 *
 * @parameters: none
 * @returns: json string
 */
std::string StringOscillatorC::ToJson() {
  std::string json_str =
      "{\n"
      "\"start_phase\":" +
      std::to_string(phase_factor) +
      ",\n"
      "\"start_frequency_factor\":" +
      std::to_string(start_frequency_factor) +
      ",\n"
      "\"start_amplitude_factor\":" +
      std::to_string(start_amplitude_factor) +
      ",\n"
      "\"sustain_factor\":" +
      std::to_string(non_sustain_factor) +
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
void StringOscillatorC::AmendGain(const double& factor) {
  start_amplitude_factor = std::clamp<double>(start_amplitude_factor * factor, 0.0, 1.0);
}

std::string StringOscillatorC::ToCsv() {
  std::string csv_str = "";
  csv_str += std::to_string(start_amplitude_factor) + "," + std::to_string(start_frequency_factor) + "," +
             std::to_string(phase_factor) + "," + std::to_string(non_sustain_factor) + "," +
             std::to_string(amplitude_decay_factor) + "," + std::to_string(amplitude_attack_factor) + "," +
             std::to_string(frequency_decay_factor) + "," + std::to_string(base_frequency_coupled);
  return csv_str;
}

/*
 * Returns a mutated version of the string, each parameter of the string
 * sound only has a 50% likelihood of being mutated.
 * @parameters: severity(determines the severity of the mutation)
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::TuneString(uint8_t severity) {
  const double sev_factor = static_cast<double>(severity) / 255.0;
  std::uniform_real_distribution<> real_distr(-sev_factor, sev_factor);
  const double phase = phase_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double start_frequency_factor = start_frequency_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double amplitude_factor = start_amplitude_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double non_sustain_factor = non_sustain_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double amplitude_decay = amplitude_decay_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double amplitude_attack = amplitude_attack_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const double frequency_decay = frequency_decay_factor + (real_distr(rand_eng) > 0) ? real_distr(rand_eng) : 0;
  const bool is_coupled = (real_distr(rand_eng) < 0.95);

  return std::make_unique<StringOscillatorC>(phase, start_frequency_factor, amplitude_factor, non_sustain_factor,
                                             amplitude_decay, amplitude_attack, frequency_decay, is_coupled);
}

static std::mt19937 stat_rand_eng = std::mt19937(std::random_device {}());
/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::CreateUntunedString(bool is_coupled) {
  std::uniform_real_distribution<> real_distr(0, 1);  // define the range.

  const double phase = real_distr(stat_rand_eng);  // Maps to 0 to TAU
  const double freq_factor = real_distr(stat_rand_eng);  // Maps to 0 to max_uncoupled_frequency_factor  or max max_coupled_frequency_factor
  const double amplitude_factor = real_distr(stat_rand_eng);    // Maps to 0 to 1
  const double non_sustain_factor = real_distr(stat_rand_eng);  // Maps to min_amplitude_decay_factor to 1;
  const double amplitude_decay = real_distr(stat_rand_eng);     // Maps to min_amplitude_decay_factor to 1;
  const double amplitude_attack = real_distr(stat_rand_eng);    // Maps to 0 to max Attack rate;
  const double frequency_decay = real_distr(stat_rand_eng);     // Maps to min_amplitude_decay_factor to 1;

  return std::make_unique<StringOscillatorC>(phase, freq_factor, amplitude_factor, non_sustain_factor, amplitude_decay,
                                             amplitude_attack, frequency_decay, is_coupled);
}

/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::CreateStringFromCsv(const std::string& csv_string) {
  std::stringstream string_stream(csv_string);
  std::vector<std::string> result;
  std::string substr;
  while (string_stream.good()) {
    getline(string_stream, substr, ',');
    result.push_back(substr);
  }

  std::uniform_real_distribution<> real_distr(0, 1);  // define the range.
  double phase = real_distr(stat_rand_eng);           // Maps to 0 to TAU
  double freq_factor =
      real_distr(stat_rand_eng);  // Maps to 0 to max_uncoupled_frequency_factor  or max max_coupled_frequency_factor
  double amplitude_factor = real_distr(stat_rand_eng) / 8;  // Maps to 0 to 1
  double non_sustain_factor = real_distr(stat_rand_eng);    // Maps to min_amplitude_decay_factor to 1;
  double amplitude_decay = real_distr(stat_rand_eng);       // Maps to min_amplitude_decay_factor to 1;
  double amplitude_attack = real_distr(stat_rand_eng);      // Maps to 0 to max Attack rate;
  double frequency_decay = real_distr(stat_rand_eng);       // Maps to min_amplitude_decay_factor to 1;
  bool is_coupled = true;
  if (result.size() > 0)
    amplitude_factor = std::stod(result[0]);
  if (result.size() > 1)
    freq_factor = std::stod(result[1]);
  if (result.size() > 2)
    phase = std::stod(result[2]);
  if (result.size() > 3)
    non_sustain_factor = std::stod(result[3]);
  if (result.size() > 4)
    amplitude_decay = std::stod(result[4]);
  if (result.size() > 5)
    amplitude_attack = std::stod(result[5]);
  if (result.size() > 6)
    frequency_decay = std::stod(result[6]);
  if (result.size() > 7)
    is_coupled = std::stod(result[7]) > 0.5;

  return std::make_unique<StringOscillatorC>(phase, freq_factor, amplitude_factor, non_sustain_factor, amplitude_decay,
                                             amplitude_attack, frequency_decay, is_coupled);
}
}  // namespace oscillator
}  // namespace instrument
