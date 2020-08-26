/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * string_oscillator.h
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */
#pragma once
#ifndef INSTRUMENT_STRING_OSCILLATOR_H_
#define INSTRUMENT_STRING_OSCILLATOR_H_

#include <cmath>

#include <memory>
#include <random>
#include <string>

#include "include/common.h"

namespace instrument {
namespace oscillator {
constexpr double k_sample_increment = 1.0 / SAMPLE_RATE;            // 0.00002267573
constexpr double k_min_amp_cutoff = 0;                              // 0
constexpr double k_max_amp_cutoff = 1;                              // 1
constexpr double k_min_amp_decay_rate = 0.99999842823;              // 0.99999842823       50% at 20 second
constexpr double k_max_amp_decay_rate = 0.99921442756;              //        50% at 20 millisecond
constexpr double k_min_freq_decay_rate = 1.0;                       //                   no decay
constexpr double k_max_freq_decay_rate = 0.99999991268;             //        50% at 180 second
constexpr double k_max_amp_attack_rate = 20 * k_sample_increment;   // 0.002267573         50 ms
constexpr double k_min_amp_attack_rate = k_sample_increment / 150;  // 1.5117158e-7        150 seconds
constexpr double k_max_coupled_freq_factor = 10.0;                  // 10
constexpr double k_max_uncoupled_freq_factor = 20000.0;             // 20000
constexpr double k_min_freq_factor = 20 / 20000;                    // 0.001

class StringOscillatorC {
 public:
  StringOscillatorC(const double& initial_phase,
                    const double& frequency_factor,
                    const double& amplitude_factor,
                    const double& amplitude_decay,
                    const double& amplitude_attack,
                    const double& frequency_decay,
                    bool is_coupled);
  void PrimeString(const double& frequency, const double& velocity);
  double NextSample();
  void AmendGain(const double& factor);
  std::string ToCsv();
  std::string ToJson();
  std::unique_ptr<StringOscillatorC> TuneString(uint8_t amount);
  static std::unique_ptr<StringOscillatorC> CreateUntunedString(bool is_coupled = true);
  static std::unique_ptr<StringOscillatorC> CreateStringFromCsv(const std::string& csv_string);

  uint32_t GetSampleNumber() const { return sample_pos; }
  const double& GetFreqFactor() const { return start_frequency_factor; }
  const double& GetAmpFactor() const { return start_amplitude_factor; }
  bool IsCoupled() const { return base_frequency_coupled; }

 private:
  // Sinusoid's start definition.
  double phase_factor;
  double start_frequency_factor;
  double start_amplitude_factor;

  // Signal modification definition.
  double amplitude_attack_factor;
  double amplitude_decay_factor;
  double frequency_decay_factor;
  bool base_frequency_coupled;

  // Signal State.
  double amplitude_decay_rate;
  double frequency_decay_rate;
  double amplitude_attack_delta;
  double max_amplitude;
  double amplitude_state;
  double frequency_state;
  double base_frequency;
  double sample_pos;
  bool in_amplitude_decay;

  std::mt19937 rand_eng;

  // Create sample of a sin function for given parameters (frequency, amplitude,
  // sample rate, phase).
  //
  // f(x) = A*sin( ( f * x/N - p ) * 2*pi)
  //
  inline double SineWave() {
    const double theta = sample_pos * frequency_state + phase_factor;
    return amplitude_state * sin(theta * M_PI * 2);
  }
};
}  // namespace oscillator
}  // namespace instrument
#endif  // INSTRUMENT_STRING_OSCILLATOR_H_
