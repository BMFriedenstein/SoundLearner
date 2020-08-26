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
#pragma once
#ifndef GENETIC_TRAINER_TRAINER_H_
#define GENETIC_TRAINER_TRAINER_H_

#include <climits>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "instrument/instrument_model.h"

namespace trainer {
// Basic training framework.
class InstumentTrainerC {
 protected:
  std::string progress_location;
  std::vector<int16_t> src_audio;
  std::vector<std::unique_ptr<instrument::InstrumentModelC>> trainees;
  double src_energy;

 public:
  virtual double GetError(const std::vector<int16_t>& tgt_audio,
                          double* corr_score,
                          double* mae_score,
                          double* diff_score);
  double CrossCorrelation(const std::vector<int16_t>& tgt_audio);
  double MeanAbsoluteError(const std::vector<int16_t>& tgt_audio, double corr_factors);
  InstumentTrainerC(uint16_t num_starting_occilators,
                    uint16_t class_size,
                    const std::vector<int16_t>& src_audio,
                    const std::string& progress_location);
  virtual ~InstumentTrainerC() {}
};

// A training framework based on a genetic algorithm.
class GeneticInstumentTrainerC : public InstumentTrainerC {
 private:
  uint32_t gens_per_addition;
  uint16_t num_generations = 0;
  uint16_t gen_count = 0;

  // TODO(Brandon): replace with MIDI parser.
  double base_frequency = 440.0;
  double velocity = 1.0;

  void GeneticAlgorithm();
  void DetermineFitness();

 public:
  GeneticInstumentTrainerC(uint16_t num_starting_occilators,
                           uint16_t class_size,
                           const std::vector<int16_t>& audio,
                           const std::string& location,
                           uint32_t gens_per_addition);
  ~GeneticInstumentTrainerC() {}
  void Start(uint16_t a_num_of_generations);
};

}  // namespace trainer

#endif  // GENETIC_TRAINER_TRAINER_H_
