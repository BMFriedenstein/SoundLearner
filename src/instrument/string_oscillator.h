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

#ifndef SRC_INSTRUMENT_STRING_OSCILLATOR_H_
#define SRC_INSTRUMENT_STRING_OSCILLATOR_H_

#include <cmath>

#include <memory>
#include <string>

#include "include/common.h"

namespace instrument {
namespace oscillator {

const double MIN_AMPLITUDE_CUTOFF     = 0.0001;
const double MIN_AMPLITUDE_DECAY_RATE     = 0.99998;   // -3db in 0.1s
const double MIN_FREQUENCY_DECAY_RATE     = 0.9999998;  // -3db in 100s
const double MAX_AMPLITUDE_ATTACK_RATE    = 0.0001;
const double MAX_COUPLED_FREQUENCY_FACTOR   = 25.0;      //  30 * 1000 > 20 kHz
const double MAX_UNCOUPLED_FREQUENCY_FACTOR = 18500.0;   //  1.0 * 18.5 kHz
const double SAMPLE_INCREMENT            = 1.0 / static_cast<double>(SAMPLE_RATE);


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
  void PrimeString(const double frequency, const double velocity);
  double NextSample(const bool sustain);
  inline uint32_t GetSampleNumber() {
    return sample_pos_;
  }
  void AmendGain(const double factor);
  std::string ToCsv();
  std::string ToJson();
  std::unique_ptr<StringOscillatorC> TuneString(const uint8_t amount);
  static std::unique_ptr<StringOscillatorC> CreateUntunedString(const bool is_coupled=true);

  double GetFreqFactor() const { return start_frequency_factor_; }
  double GetAmpFactor() const { return start_amplitude_factor_; }
  double IsCoupled() const { return base_frequency_coupled_; }
 private:
  // Sinusoid's start definition.
  double phase_factor_;
  double start_frequency_factor_;
  double start_amplitude_factor_;

  // Signal modification definition.
  double non_sustain_factor_ = 1.0;
  double amplitude_attack_factor_ = 0.0;
  double amplitude_decay_factor_ = 0.0;
  double frequency_decay_factor_ = 0.0;
  bool base_frequency_coupled_ = true;

  // Signal State.
  double normal_amplitude_decay_rate_ = 0.0;
  double sutain_amplitude_decay_rate_ = 0.0;
  double frequency_decay_rate_ = 0.0;
  double amplitude_attack_delta_ = 0.0;
  double max_amplitude_ = 0.0;
  double amplitude_state_ = 0.0;
  double frequency_state_ = 0.0;
  double base_frequency_ = 0.0;
  double sample_pos_ = 0.0;
  bool in_amplitude_decay_ = false;

  // Create sample of a sin function for given parameters (frequency, amplitude,
  // sample rate, phase).
  //
  // f(x) = A*sin( ( f * x/N - p ) * 2*pi)
  //
  inline double SineWave() {
    double theta = sample_pos_ * frequency_state_ + phase_factor_;
    return amplitude_state_ * sin(theta * TAU);
  }
};
}  // namespace oscillator
}  // namespace instrument
#endif  // SRC_INSTRUMENT_STRING_OSCILLATOR_H_
