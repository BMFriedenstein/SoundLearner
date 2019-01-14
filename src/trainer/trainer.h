/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * trainer.h
 *  Created on: 04Jan 2019
 *      Author: Brandon
 */

#ifndef SRC_TRAINER_TRAINER_H_
#define SRC_TRAINER_TRAINER_H_

#include <climits>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "instrument/instrument_model.h"
#include "progress_logger.h"

namespace trainer {
// Basic training framework.
class InstumentTrainerC {
 protected:
  std::string progress_location_;
  std::vector<int16_t> src_audio_;
  logging::ProgressLogC logger;
  std::vector<std::unique_ptr<instrument::InstrumentModelC>> trainees_;
  double src_energy_ = 0;
 public:
  virtual double GetError(const std::vector<int16_t>& tgt_audio);
  double CrossCorrelation(const std::vector<int16_t>& tgt_audio);
  double MeanAbsoluteError(const std::vector<int16_t>& tgt_audio,
                           const double corr_factors);
  InstumentTrainerC(uint16_t num_starting_occilators, uint16_t class_size,
                    std::vector<int16_t>& src_audio,
                    std::string& progress_location);
  virtual ~InstumentTrainerC() {
  }
};

// A training framework based on a genetic algorithm.
class GeneticInstumentTrainerC : public InstumentTrainerC {
 private:
  uint32_t gens_per_addition_;
  uint16_t num_generations_ = 0;
  uint16_t gen_count_ = 0;

  // TODO(Brandon): replace with MIDI input.
  double base_frequency = 440.0;
  std::vector<bool> sustain;
  double velocity = 1.0;

  void GeneticAlgorithm();
  void DetermineFitness();

 public:
  GeneticInstumentTrainerC(uint16_t num_starting_occilators,
                           uint16_t class_size, std::vector<int16_t>& src_audio,
                           std::string& progress_location,
                           uint32_t gens_per_addition);
  ~GeneticInstumentTrainerC() {
  }
  void Start(const uint16_t a_num_of_generations);
};

}  // namespace trainer

#endif  // SRC_TRAINER_TRAINER_H_
