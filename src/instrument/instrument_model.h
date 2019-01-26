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
#ifndef SRC_INSTRUMENT_INSTRUMENT_MODEL_H_
#define SRC_INSTRUMENT_INSTRUMENT_MODEL_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "instrument/string_oscillator.h"

namespace instrument {

enum SortType{ none, amplitude, frequency };
class InstrumentModelC {
 public:
  double error_score_;
  double mae_score_;
  double corr_score_;
  double diff_score_;

  bool score_is_cached_ = false;
  InstrumentModelC(uint16_t num_strings,
                   std::string instrument_name);
  InstrumentModelC(uint16_t num_coupled_strings,
                   uint16_t num_uncoupled_strings,
                   std::string instrument_name);
  std::string GetName() {
    return name_;
  }

  void AddTunedString(const oscillator::StringOscillatorC a_tuned_string);
  void AddUntunedString(const bool is_uncoupled=false);

  std::string ToCsv(SortType sort_type = none);
  std::string ToJson(SortType sort_type = none);
  std::vector<double> GenerateSignal(const double velocity,
                                     const double frequency,
                                     const uint32_t num_of_samples,
                                     std::vector<bool>& sustain);
  std::vector<int16_t> GenerateIntSignal(const double velocity,
                                         const double frequency,
                                         const uint32_t num_of_samples,
                                         std::vector<bool>& sustain);

  std::unique_ptr<InstrumentModelC> TuneInstrument(const uint8_t amount);
  void AmendGain(const double factor);

  bool operator<(const InstrumentModelC& other) const {
    return error_score_ < other.error_score_;
  }

  // TODO(BRANDON) for player application:
  // void PrimeNotePlayed(double frequency, double velocity);
  // double GenerateNextSample(bool sustain);
 private:
  const uint16_t max_strings_ = 1000;
  std::vector<std::unique_ptr<oscillator::StringOscillatorC>> sound_strings_;
  std::string name_;

  inline void SortStringsByFreq(){
    std::sort(sound_strings_.begin(),
              sound_strings_.end(),
              [](const std::unique_ptr<oscillator::StringOscillatorC>& a_osc,
                  const std::unique_ptr<oscillator::StringOscillatorC>& b_osc) -> bool
              {
                if( a_osc->IsCoupled() && !b_osc->IsCoupled()) {return true;}
                else if(a_osc->IsCoupled() == b_osc->IsCoupled()) {
                  return a_osc->GetFreqFactor() > b_osc->GetFreqFactor();
                }
                else {
                  return false;
                }
              });
  }

  inline void SortStringsByAmplitude(){
    std::sort(sound_strings_.begin(),
              sound_strings_.end(),
              [](const std::unique_ptr<oscillator::StringOscillatorC>& a_osc,
                 const std::unique_ptr<oscillator::StringOscillatorC>& b_osc) -> bool
              {
                if( a_osc->IsCoupled() && !b_osc->IsCoupled()) {return true;}
                else if(a_osc->IsCoupled() == b_osc->IsCoupled()) {
                  return a_osc->GetAmpFactor() > b_osc->GetAmpFactor();
                }
                else {
                  return false;
                }
              });
  }
};
}  // namespace instrument

#endif  // SRC_INSTRUMENT_INSTRUMENT_MODEL_H_
