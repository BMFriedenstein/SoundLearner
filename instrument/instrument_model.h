/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * instrument_model.h
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */
#pragma once
#ifndef INSTRUMENT_INSTRUMENT_MODEL_H_
#define INSTRUMENT_INSTRUMENT_MODEL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "instrument/string_oscillator.h"

namespace instrument {
enum SortType { none, amplitude, frequency };
class InstrumentModelC {
 public:
  static constexpr uint16_t k_max_strings = 1000;
  double error_score;
  double mae_score;
  double corr_score;
  double diff_score;

  bool score_is_cached;
  InstrumentModelC(const std::vector<std::string>& csv_string, const  std::string& a_name);
  InstrumentModelC(uint16_t num_strings, const std::string& instrument_name);
  InstrumentModelC(uint16_t num_coupled_strings, uint16_t num_uncoupled_strings, const std::string& instrument_name);
  const std::string& GetName() const { return name; }

  void AddTunedString(const oscillator::StringOscillatorC&& a_tuned_string);
  void AddUntunedString(bool is_uncoupled = false);

  std::string ToCsv(SortType sort_type = none);
  std::string ToJson(SortType sort_type = none);
  std::vector<double> GenerateSignal(double velocity,
                                     double frequency,
                                     uint32_t num_of_samples,
                                     std::vector<bool>* sustain);
  std::vector<int16_t> GenerateIntSignal(double velocity,
                                         double frequency,
                                         uint32_t num_of_samples,
                                         std::vector<bool>* sustain,
                                         bool* has_distorted_out,
                                         bool return_on_distort = true);

  std::unique_ptr<InstrumentModelC> TuneInstrument(const uint8_t amount);
  void AmendGain(const double factor);

  bool operator<(const InstrumentModelC& other) const { return error_score < other.error_score; }

  // TODO(BRANDON) for player application:
  // void PrimeNotePlayed(double frequency, double velocity);
  // double GenerateNextSample(bool sustain);
 private:
  std::vector<std::unique_ptr<oscillator::StringOscillatorC>> sound_strings;
  std::string name;

  inline void SortStringsByFreq() {
    std::sort(sound_strings.begin(), sound_strings.end(),
              [](const std::unique_ptr<oscillator::StringOscillatorC>& a_osc,
                 const std::unique_ptr<oscillator::StringOscillatorC>& b_osc) -> bool {
                if (a_osc->IsCoupled() && !b_osc->IsCoupled()) {
                  return true;
                } else if (a_osc->IsCoupled() == b_osc->IsCoupled()) {
                  return a_osc->GetFreqFactor() > b_osc->GetFreqFactor();
                } else {
                  return false;
                }
              });
  }

  inline void SortStringsByAmplitude() {
    std::sort(sound_strings.begin(), sound_strings.end(),
              [](const std::unique_ptr<oscillator::StringOscillatorC>& a_osc,
                 const std::unique_ptr<oscillator::StringOscillatorC>& b_osc) -> bool {
                if (a_osc->IsCoupled() && !b_osc->IsCoupled()) {
                  return true;
                } else if (a_osc->IsCoupled() == b_osc->IsCoupled()) {
                  return a_osc->GetAmpFactor() > b_osc->GetAmpFactor();
                } else {
                  return false;
                }
              });
  }
};
}  // namespace instrument

#endif  // INSTRUMENT_INSTRUMENT_MODEL_H_
