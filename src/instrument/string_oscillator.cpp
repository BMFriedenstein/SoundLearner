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

#include <random>
namespace instrument {
namespace oscillator {
StringOscillatorC::StringOscillatorC(double phase, double freq_factor,
                                     double amp_factor, double sus_factor,
                                     double amp_decay, double amp_attack,
                                     double freq_decay, double freq_attack) {
  // Parameter limits.
  if (amp_decay > 1) {
    amp_decay = 1;
  }
  if (amp_decay < 0) {
    amp_decay = 0;
  }
  if (freq_decay > 1) {
    amp_decay = 1;
  }
  if (freq_decay < 0) {
    amp_decay = 0;
  }
  if (amp_attack > amp_factor) {
    amp_attack = amp_factor;
  }
  if (amp_attack < 0) {
    amp_attack = 0;
  }
  if (freq_attack > freq_factor) {
    freq_attack = freq_factor;
  }
  if (freq_attack < 0) {
    freq_attack = 0;
  }
  if (freq_factor < 0) {
    freq_factor = 0;
  }
  if (freq_factor > 400) {
    freq_factor = 400;
  }
  if (amp_factor < 0) {
    amp_factor = 0;
  }
  if (amp_factor > 1) {
    amp_factor = 1;
  }
  if (sustain_factor_ < 0) {
    sustain_factor_ = 0;
  }

  start_phase_ = phase;
  start_frequency_factor_ = freq_factor;
  start_amplitude_factor_ = amp_factor;
  sustain_factor_ = sus_factor;
  amplitude_decay_rate_ = amp_decay;
  amplitude_attack_delta_ = amp_attack;
  frequency_decay_rate_ = freq_decay;
  frequency_attack_delta_ = freq_attack;
}

/*
 * Parse information required to generate a signal.
 *
 * @parameters: frequency (The base note), velocity( how hard of note was pressed  form 0-1)
 * @returns: void
 */
void StringOscillatorC::PrimeString(const double freq, const double velocity) {
  max_amplitude_ = velocity * start_amplitude_factor_;
  max_frequency_ = freq * start_frequency_factor_;
  if (max_frequency_ > SAMPLE_RATE / 2) {
    max_frequency_ = SAMPLE_RATE / 2;
  }
  amplitude_state_ = 0;
  base_frequency_ = freq;
  frequency_state_ = base_frequency_;
  sample_num_ = 0;
  in_amplitude_decay = false;
  in_frequency_decay = false;
}

/*
 * Generate the value of the next sample of the signal.
 *
 * @parameters: sustain (Is the note still being pressed)
 * @returns: next value of the signal float
 */
double StringOscillatorC::NextSample(const bool sustain) {
  // Calculate amplitude state
  if (!in_amplitude_decay) {
    amplitude_state_ = amplitude_state_ + amplitude_attack_delta_;
  } else {
    double decay = (sustain ? sustain_factor_ : 1) * amplitude_decay_rate_;
    amplitude_state_ = amplitude_state_ * (decay > 1 ? 1 : decay);
  }
  if (amplitude_state_ < 0) {
    amplitude_state_ = 0;
  } else if (amplitude_state_ > max_amplitude_) {
    in_amplitude_decay = true;
    amplitude_state_ = max_amplitude_;
  }

  // Calculate frequency state.
  if (!in_frequency_decay) {
    frequency_state_ = frequency_state_ + frequency_attack_delta_;
  } else {
    double decay = (sustain ? sustain_factor_ : 1) * frequency_decay_rate_;
    frequency_state_ = frequency_state_ * (decay > 1 ? 1 : decay);
  }
  if (frequency_state_ < 0) {
    frequency_state_ = 0;
  }
  if (frequency_state_ > max_frequency_) {
    in_frequency_decay = true;
    frequency_state_ = max_frequency_;
  }

  // generate sample.
  double sample_val = SineWave();

  // increment sample number.
  sample_num_++;

  return sample_val;
}

/*
 * Create a JSON representation of the SoundString.
 *
 * @parameters: none
 * @returns: json string
 */
std::string StringOscillatorC::ToJson() {
  std::string json_str = "{\n";
  json_str += "\"start_phase\":" + std::to_string(start_phase_) + ",\n";
  json_str += "\"start_frequency_factor\":"
      + std::to_string(start_frequency_factor_) + ",\n";
  json_str += "\"start_amplitude_factor\":"
      + std::to_string(start_amplitude_factor_) + ",\n";
  json_str += "\"sustain_factor\":" + std::to_string(sustain_factor_) + ",\n";
  json_str += "\"amp_decay_rate\":" + std::to_string(amplitude_decay_rate_)
      + ",\n";
  json_str += "\"amp_attack_delta\":" + std::to_string(amplitude_attack_delta_)
      + ",\n";
  json_str += "\"freq_decay_rate\":" + std::to_string(frequency_decay_rate_)
      + ",\n";
  json_str += "\"freq_attack_delta\":" + std::to_string(frequency_attack_delta_)
      + "\n";
  json_str += "}\n";
  return json_str;
}

/*
 * Returns a mutated version of the string, each parameter of the string sound only has a 50% likelihood of being mutated
 * @parameters: severity(determines the severity of the mutation)
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::TuneString(
    const uint8_t severity) {
  float sev_factor = severity / 255L;

  std::random_device random_device;   // obtain a random number from hardware
  std::mt19937 eng(random_device());  // seed the generator
  std::uniform_real_distribution<> real_distr(-sev_factor, sev_factor);
  double phase =
      start_phase_ + (real_distr(eng) > 0) ? 2 * PI * real_distr(eng) : 0;
  double freq_factor =
      start_frequency_factor_ + (real_distr(eng) > 0) ?
          10 * real_distr(eng) : 0;
  double amp_factor =
      start_amplitude_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double sus_factor =
      sustain_factor_ + (real_distr(eng) > 0) ? 0.2 * real_distr(eng) : 0;
  double amp_decay =
      amplitude_decay_rate_ + (real_distr(eng) > 0) ?
          0.0001 * real_distr(eng) : 0;
  double amp_attack =
      amplitude_attack_delta_ + (real_distr(eng) > 0) ?
          0.01 * real_distr(eng) : 0;
  double freq_decay =
      frequency_decay_rate_ + (real_distr(eng) > 0) ? 0.1 * real_distr(eng) : 0;
  double freq_attack =
      frequency_attack_delta_ + (real_distr(eng) > 0) ?
          1000 * real_distr(eng) : 0;

  auto tuned_string = std::unique_ptr<StringOscillatorC> {
      new StringOscillatorC(phase, freq_factor, amp_factor, sus_factor,
                            amp_decay, amp_attack, freq_decay, freq_attack) };

  return tuned_string;
}

/*
 * Generates a new completely randomized soundstring
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::CreateUntunedString() {
  std::random_device random_device;  // obtain a random number from hardware
  std::mt19937 eng(random_device());  // seed the generator
  std::uniform_real_distribution<> real_distr(0, 1);  // define the range
  double phase = 2 * PI * real_distr(eng);
  double freq_factor = 20 * real_distr(eng);
  double amp_factor = real_distr(eng);
  double sus_factor = 0.99999 + 0.0000115 * real_distr(eng);
  double amp_decay = 0.99995 + 0.00005 * real_distr(eng);
  double amp_attack = 0.01 * real_distr(eng);
  double freq_decay = 0.9999 + 0.0001 * real_distr(eng);
  double freq_attack = 1000 * real_distr(eng);

  auto untuned_string = std::unique_ptr<StringOscillatorC> {
      new StringOscillatorC(phase, freq_factor, amp_factor, sus_factor,
                            amp_decay, amp_attack, freq_decay, freq_attack) };

  return untuned_string;
}
}  // namespace oscillator
}  // namespace instrument
