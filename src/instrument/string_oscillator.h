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
class StringOscillatorC {
 public:
  StringOscillatorC(const double phase, const double freq_factor,
                    const double amp_factor, const double sus_factor,
                    const double amp_decay, const double amp_attack,
                    const double freq_decay, const double freq_attack);
  void PrimeString(const double freq, const double velocity);
  double NextSample(const bool sustain);
  inline uint32_t GetSampleNumber() {
    return sample_num_;
  }
  std::string ToJson();
  std::unique_ptr<StringOscillatorC> TuneString(const uint8_t amount);
  static std::unique_ptr<StringOscillatorC> CreateUntunedString();

 private:
  // Sinusoid's start definition.
  double start_phase_;
  double start_frequency_factor_;
  double start_amplitude_factor_;

  // Signal modification definition.
  double sustain_factor_;
  double amplitude_attack_delta_ = 0;
  double amplitude_decay_rate_ = 0;
  double frequency_attack_delta_ = 0;
  double frequency_decay_rate_ = 0;

  // Signal State.
  double max_amplitude_ = 0;
  double max_frequency_ = 0;
  double amplitude_state_ = 0;
  double frequency_state_ = 0;
  double base_frequency_ = 0;
  uint32_t sample_num_ = 0;
  bool in_amplitude_decay = false;
  bool in_frequency_decay = false;

  // Create sample of a sin function for given parameters (frequency, amplitude,
  // sample rate, phase)
  //
  // f(x) = A*sin( (2*x*pi*f/N) -p)
  //
  inline double SineWave() {
    double theta = ((static_cast<double>(sample_num_)
        / static_cast<double>(SAMPLE_RATE)) * 2.0 * PI * frequency_state_)
        - start_phase_;
    return amplitude_state_ * sin(theta);
  }
};
}  // namespace oscillator
}  // namespace instrument
#endif  // SRC_INSTRUMENT_STRING_OSCILLATOR_H_
