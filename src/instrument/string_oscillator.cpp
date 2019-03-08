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
StringOscillatorC::StringOscillatorC(double initial_phase,
                                     double frequency_factor,
                                     double amplitude_factor,
                                     double non_sustain_factor,
                                     double amplitude_decay,
                                     double amplitude_attack,
                                     double frequency_decay,
                                     bool is_coupled) {
  // Parameter limits.
  if (amplitude_decay > 1.0) { amplitude_decay = 1.0; }
  if (amplitude_decay < 2 * MIN_AMPLITUDE_CUTOFF) {
    amplitude_decay = 2 * MIN_AMPLITUDE_CUTOFF;
  }
  if (frequency_decay > (1.0)) { frequency_decay = (1.0); }
  if (frequency_decay < 0.0) { frequency_decay = 0.0; }
  if (amplitude_attack > 1.0) {  amplitude_attack = 1.0; }
  if (amplitude_attack < 2 * MIN_AMPLITUDE_CUTOFF) {
    amplitude_attack = 2 * MIN_AMPLITUDE_CUTOFF;
  }
  if (frequency_factor < 0.0) { frequency_factor = 0.0; }
  if (frequency_factor > 1.0) { frequency_factor = 1.0; }
  if (amplitude_factor < 0.0) { amplitude_factor = 0.0; }
  if (amplitude_factor > 1.0) { amplitude_factor = 1.0; }
  if (non_sustain_factor < 0.0) { non_sustain_factor = 0.0; }
  if (non_sustain_factor > 1.0) { non_sustain_factor = 1.0; }
  if (initial_phase < 0.0) { initial_phase = 0.0; }
  if (initial_phase > 1.0) { initial_phase = 1.0; }

  phase_factor_ = initial_phase;
  start_frequency_factor_ = frequency_factor;
  start_amplitude_factor_ = amplitude_factor;
  non_sustain_factor_ = non_sustain_factor;
  amplitude_decay_factor_ = amplitude_decay;
  amplitude_attack_factor_ = amplitude_attack;
  frequency_decay_factor_ = frequency_decay;
  base_frequency_coupled_ = is_coupled;
}

/*
 * Parse information required to generate a signal.
 *
 * @parameters: frequency (The base note), velocity( how hard of note was pressed  form 0-1)
 * @returns: void
 */
void StringOscillatorC::PrimeString(const double freq, const double velocity) {
  if (start_amplitude_factor_ < MIN_AMPLITUDE_CUTOFF ||
      amplitude_attack_factor_ < MIN_AMPLITUDE_CUTOFF * MAX_AMPLITUDE_ATTACK_RATE) {
    return;
  }
  max_amplitude_ = velocity * start_amplitude_factor_;
  amplitude_state_ = 0;
  base_frequency_ = freq;
  sample_pos_ = 0;
  in_amplitude_decay_ = false;

  if (base_frequency_coupled_) {
    frequency_state_ = base_frequency_ * MAX_COUPLED_FREQUENCY_FACTOR * start_frequency_factor_;
  }
  else {
    frequency_state_ = MAX_UNCOUPLED_FREQUENCY_FACTOR * start_frequency_factor_;
  }
  amplitude_attack_delta_ = amplitude_attack_factor_ * MAX_AMPLITUDE_ATTACK_RATE;
  normal_amplitude_decay_rate_ = 1 - (1 - MIN_AMPLITUDE_DECAY_RATE) * amplitude_decay_factor_;
  sutain_amplitude_decay_rate_ = normal_amplitude_decay_rate_
                                 * (1 - (1 - MIN_AMPLITUDE_DECAY_RATE) * non_sustain_factor_);
  frequency_decay_rate_ = 1 - (1 - MIN_FREQUENCY_DECAY_RATE) * frequency_decay_factor_;

  if (frequency_state_ > SAMPLE_RATE / 2) {
    frequency_state_ = SAMPLE_RATE / 2;
  }
}

/*
 * Generate the value of the next sample of the signal.
 *
 * @parameters: sustain (Is the note still being pressed)
 * @returns: next value of the signal float
 */
double StringOscillatorC::NextSample(const bool sustain) {
  if (start_amplitude_factor_ < MIN_AMPLITUDE_CUTOFF ||
      amplitude_attack_factor_  < MIN_AMPLITUDE_CUTOFF * MAX_AMPLITUDE_ATTACK_RATE) {
    return 0.0;
  }

  // Calculate amplitude state
  if (!in_amplitude_decay_) {
    amplitude_state_ = amplitude_state_ + amplitude_attack_delta_;
  }
  else {
    if (sustain) {
      amplitude_state_ = amplitude_state_ * normal_amplitude_decay_rate_;
    }
    else {
      amplitude_state_ = amplitude_state_ * sutain_amplitude_decay_rate_;
    }
  }

  if (amplitude_state_ >= max_amplitude_) {
    in_amplitude_decay_ = true;
    amplitude_state_ = max_amplitude_;
  }

  // Calculate frequency state.
  frequency_state_ = frequency_state_ * frequency_decay_rate_;

  if(frequency_state_ > 15000){

  }
  // generate sample.
  double sample_val = SineWave();
  sample_pos_ += SAMPLE_INCREMENT;

  // Apply some filtering
  if(frequency_state_ > 20000){
    sample_val = 0;
  }
  else if (frequency_state_ > 15000) {
    sample_val = sample_val * (1 - 0.0005*frequency_state_ + 9);
  }
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
  json_str += "\"start_phase\":" + std::to_string(phase_factor_) + ",\n";
  json_str += "\"start_frequency_factor\":" + std::to_string(start_frequency_factor_) + ",\n";
  json_str += "\"start_amplitude_factor\":" + std::to_string(start_amplitude_factor_) + ",\n";
  json_str += "\"sustain_factor\":" + std::to_string(non_sustain_factor_) + ",\n";
  json_str += "\"amp_decay_rate\":" + std::to_string(amplitude_decay_factor_) + ",\n";
  json_str += "\"amp_attack_delta\":" + std::to_string(amplitude_attack_factor_) + ",\n";
  json_str += "\"freq_decay_rate\":" + std::to_string(frequency_decay_factor_) + ",\n";
  json_str += "\"base_frequency_coupled\":" + std::to_string(base_frequency_coupled_) + "\n";
  json_str += "}";
  return json_str;
}
void StringOscillatorC::AmendGain(const double factor) {
   start_amplitude_factor_ *= factor;
   if (start_amplitude_factor_ < 0.0) { start_amplitude_factor_ = 0.0; }
   if (start_amplitude_factor_ > 1.0) { start_amplitude_factor_ = 1.0; }
}

std::string StringOscillatorC::ToCsv(){
  std::string csv_str = "";
  csv_str +=  std::to_string(start_amplitude_factor_) + "," +
              std::to_string(start_frequency_factor_) + "," +
              std::to_string(phase_factor_) + "," +
              std::to_string(non_sustain_factor_) + "," +
              std::to_string(amplitude_decay_factor_) + "," +
              std::to_string(amplitude_attack_factor_) + "," +
              std::to_string(frequency_decay_factor_) + "," +
              std::to_string(base_frequency_coupled_);
  return csv_str;
}

/*
 * Returns a mutated version of the string, each parameter of the string
 * sound only has a 50% likelihood of being mutated.
 * @parameters: severity(determines the severity of the mutation)
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::TuneString(
    const uint8_t severity) {
  double sev_factor = static_cast<double>(severity) / 255.0;

  std::random_device random_device;   // obtain a random number from hardware.
  std::mt19937 eng(random_device());  // seed the generator.
  std::uniform_real_distribution<> real_distr(-sev_factor, sev_factor);
  double phase = phase_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double start_frequency_factor = start_frequency_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double amplitude_factor = start_amplitude_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double non_sustain_factor = non_sustain_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double amplitude_decay = amplitude_decay_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double amplitude_attack = amplitude_attack_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;
  double frequency_decay = frequency_decay_factor_ + (real_distr(eng) > 0) ? real_distr(eng) : 0;

  // Is frequency factor linked to base frequency
  bool is_coupled = (real_distr(eng) < 0.95);

  auto tuned_string = std::unique_ptr<StringOscillatorC> {
      new StringOscillatorC(phase, start_frequency_factor, amplitude_factor,
                            non_sustain_factor, amplitude_decay,
                            amplitude_attack, frequency_decay, is_coupled) };

  return tuned_string;
}

/*
 * Generates a new completely randomized SoundString oscillator.
 * @parameters: none
 * @returns: Sound string pointer
 */
std::unique_ptr<StringOscillatorC> StringOscillatorC::CreateUntunedString(const bool is_coupled) {
  std::random_device random_device;     // obtain a random number from hardware.
  std::mt19937 eng(random_device());                  // seed the generator.
  std::uniform_real_distribution<> real_distr(0, 1);  // define the range.

  double phase = real_distr(eng);               // Maps to 0 to TAU
  double freq_factor = real_distr(eng);         // Maps to 0 to max_uncoupled_frequency_factor  or max max_coupled_frequency_factor
  double amplitude_factor = real_distr(eng);    // Maps to 0 to 1
  double non_sustain_factor = real_distr(eng);  // Maps to min_amplitude_decay_factor to 1;
  double amplitude_decay = real_distr(eng);     // Maps to min_amplitude_decay_factor to 1;
  double amplitude_attack = real_distr(eng);    // Maps to 0 to max Attack rate;
  double frequency_decay = real_distr(eng);     // Maps to min_amplitude_decay_factor to 1;

  auto untuned_string = std::unique_ptr<StringOscillatorC> {
    new StringOscillatorC(phase,
                          freq_factor,
                          amplitude_factor,
                          non_sustain_factor,
                          amplitude_decay,
                          amplitude_attack,
                          frequency_decay,
                          is_coupled)
  };

  return untuned_string;
}
}  // namespace oscillator
}  // namespace instrument
