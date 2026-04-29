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
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
namespace instrument {
namespace oscillator {
namespace {
constexpr std::array<double, 7> kFrequencyAnchors = {
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
};
double g_untuned_frequency_factor_min = 0.0;
double g_untuned_frequency_factor_max = 1.0;

double ClampRenderedFrequency(double frequency) {
  return std::clamp(frequency, 0.0, k_max_rendered_frequency);
}

template <std::size_t N>
double QuantizedFrequencyFactor(double normalized_factor, const std::array<double, N> &anchors, double detune_ratio) {
  const double clamped = std::clamp(normalized_factor, 0.0, 1.0);
  const double scaled = clamped * static_cast<double>(anchors.size());
  const std::size_t anchor_index = std::min<std::size_t>(static_cast<std::size_t>(scaled), anchors.size() - 1);
  const double local = std::clamp(scaled - static_cast<double>(anchor_index), 0.0, 1.0);
  const double detune = 1.0 + ((local - 0.5) * 2.0 * detune_ratio);
  return anchors[anchor_index] * detune;
}

double StructuredFrequencyFactor(double normalized_factor, bool is_coupled) {
  if (is_coupled) {
    return QuantizedFrequencyFactor(normalized_factor, kFrequencyAnchors, k_coupled_detune_ratio);
  }
  return QuantizedFrequencyFactor(normalized_factor, kFrequencyAnchors, k_uncoupled_detune_ratio);
}
} // namespace

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
  const double frequency_factor = StructuredFrequencyFactor(start_frequency_factor, base_frequency_coupled);
  amplitude_state = 0.0;
  sample_pos = 0U;
  in_amplitude_decay = false;
  max_amplitude = velocity * amplitude_factor;
  base_frequency = freq;
  frequency_state = ClampRenderedFrequency(base_frequency * frequency_factor);
  amplitude_attack_delta = amplitude_attack * max_amplitude;
  amplitude_decay_rate = amplitude_decay;
  frequency_decay_rate = frequency_decay;
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
    frequency_state = ClampRenderedFrequency(frequency_state * frequency_decay_rate);
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

void StringOccilator::SetUntunedFrequencyFactorRange(double minimum, double maximum) {
  const double clamped_minimum = std::clamp(minimum, 0.0, 1.0);
  const double clamped_maximum = std::clamp(maximum, 0.0, 1.0);
  g_untuned_frequency_factor_min = std::min(clamped_minimum, clamped_maximum);
  g_untuned_frequency_factor_max = std::max(clamped_minimum, clamped_maximum);
}

/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOccilator> StringOccilator::CreateUntunedString(bool is_coupled) {
  std::uniform_real_distribution<> real_distr(0, 1); // define the range.
  std::uniform_real_distribution<> frequency_distr(g_untuned_frequency_factor_min, g_untuned_frequency_factor_max);

  const double phase = real_distr(stat_rand_eng);            // Maps to 0 to TAU
  const double freq_factor = frequency_distr(stat_rand_eng); // Maps to the structured octave-anchor frequency ladder.
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
  const double amplitude_decay = result.size() > 3 ? std::stod(result[3]) : real_distr(stat_rand_eng);
  const double amplitude_attack = result.size() > 4 ? std::stod(result[4]) : real_distr(stat_rand_eng);
  const double frequency_decay = result.size() > 5 ? std::stod(result[5]) : real_distr(stat_rand_eng);
  const bool is_coupled = result.size() > 6 ? std::stod(result[6]) > 0.5 : true;

  return std::make_unique<StringOccilator>(phase, freq_factor, amplitude_factor, amplitude_decay, amplitude_attack, frequency_decay, is_coupled);
}
} // namespace oscillator
} // namespace instrument
