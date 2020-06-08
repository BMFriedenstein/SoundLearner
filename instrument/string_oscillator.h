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
constexpr double SAMPLE_INCREMENT               = 1.0 / SAMPLE_RATE; // 0.00002267573
constexpr double MIN_AMPLITUDE_CUTOFF           = 0;                                   // 0
constexpr double MAX_AMPLITUDE_CUTOFF           = 1;                                   // 1
const double MIN_AMPLITUDE_DECAY_RATE           = std::pow(0.5,SAMPLE_INCREMENT/20);   // 0.99999842823       50% at 20 second
const double MAX_AMPLITUDE_DECAY_RATE           = std::pow(0.5,SAMPLE_INCREMENT/0.02); // 0.99921442756       50% at 20 millisecond
constexpr double MIN_FREQUENCY_DECAY_RATE       = 1;                                   // 1                   no decay
const double MAX_FREQUENCY_DECAY_RATE           = std::pow(0.5,SAMPLE_INCREMENT/180);  // 0.99999991268       50% at 180 second
constexpr double MAX_AMPLITUDE_ATTACK_RATE      = 20*SAMPLE_INCREMENT;                 // 0.002267573         50 ms
constexpr double MIN_AMPLITUDE_ATTACK_RATE      = SAMPLE_INCREMENT/150;                // 1.5117158e-7        150 seconds
constexpr double MAX_COUPLED_FREQUENCY_FACTOR   = 10.0;                                // 10
constexpr double MAX_UNCOUPLED_FREQUENCY_FACTOR = 20000.0;                             // 20000
constexpr double MIN_FREQUENCY_FACTOR           = 20/20000;                            // 0.001

class StringOscillatorC {
 public:
  StringOscillatorC(double initial_phase,
                    double frequency_factor,
                    double amplitude_factor,
                    double non_sustain_factor,
                    double amplitude_decay,
                    double amplitude_attack,
                    double frequency_decay,
                    bool is_coupled);
  void PrimeString(double frequency, double velocity);
  double NextSample(bool sustain);
  void AmendGain(double factor);
  std::string ToCsv();
  std::string ToJson();
  std::unique_ptr<StringOscillatorC> TuneString(uint8_t amount);
  static std::unique_ptr<StringOscillatorC> CreateUntunedString(bool is_coupled=true);
  static std::unique_ptr<StringOscillatorC> CreateStringFromCsv(const std::string& csv_string);

  uint32_t GetSampleNumber() const { return sample_pos; }
  double GetFreqFactor() const { return start_frequency_factor; }
  double GetAmpFactor() const { return start_amplitude_factor; }
  double IsCoupled() const { return base_frequency_coupled; }

 private:
  // Sinusoid's start definition.
  double phase_factor;
  double start_frequency_factor;
  double start_amplitude_factor;

  // Signal modification definition.
  double non_sustain_factor = 1.0;
  double amplitude_attack_factor = 0.0;
  double amplitude_decay_factor = 0.0;
  double frequency_decay_factor = 0.0;
  bool base_frequency_coupled = true;

  // Signal State.
  double normal_amplitude_decay_rate = 0.0;
  double sutain_amplitude_decay_rate = 0.0;
  double frequency_decay_rate = 0.0;
  double amplitude_attack_delta = 0.0;
  double max_amplitude = 0.0;
  double amplitude_state = 0.0;
  double frequency_state = 0.0;
  double base_frequency = 0.0;
  double sample_pos = 0.0;
  bool in_amplitude_decay = false;

  std::mt19937 rand_eng = std::mt19937(std::random_device{}());

  // Create sample of a sin function for given parameters (frequency, amplitude,
  // sample rate, phase).
  //
  // f(x) = A*sin( ( f * x/N - p ) * 2*pi)
  //
  inline double SineWave() {
    const double theta = sample_pos * frequency_state + phase_factor;
    return amplitude_state * sin(theta * M_PI*2);
  }
};
}  // namespace oscillator
}  // namespace instrument
#endif  // INSTRUMENT_STRING_OSCILLATOR_H_
