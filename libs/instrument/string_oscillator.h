/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * sound_string.h
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */

#ifndef STRINGSOUNDS_LIBS_INSTRUMENT_STRING_OSCILLATOR_H_
#define STRINGSOUNDS_LIBS_INSTRUMENT_STRING_OSCILLATOR_H_

#include <math.h>
#include <memory>
#include <string>

#include "include/common.h"

namespace instrument {
namespace oscillator {
class StringOscillatorC {
public:
    StringOscillatorC(
            const double phase,
            const double freq_factor,
            const double amp_factor,
            const double sus_factor,
            const double amp_decay,
            const double amp_attack,
            const double freq_decay,
            const double freq_attack
    );
    void PrimeString(const double freq, const double velocity);
    double NextSample(const bool sustain);
    inline uint32_t GetSampleNumber() { return sample_num_; }
    std::string ToJson();
    std::unique_ptr<StringOscillatorC> TuneString(const uint8_t amount);
    static std::unique_ptr<StringOscillatorC> CreateUntunedString();
private:
    // Sinusoid's start definition
    double start_phase_;
    double start_frequency_factor_;
    double start_amplitude_factor_;

    // Signal modification definition
    double sustain_factor_;
    double amplitude_attack_delta_     = 0;
    double amplitude_decay_rate_       = 0;
    double frequency_attack_delta_    = 0;
    double frequency_decay_rate_      = 0;

    // Signal State
    double max_amplitude_      = 0;
    double max_frequency_     = 0;
    double amplitude_state_    = 0;
    double frequency_state_   = 0;
    double base_frequency_    = 0;
    uint32_t sample_num_  = 0;
    bool in_amplitude_decay   = false;
    bool in_frequency_decay  = false;

    inline double SineWave() {
        return amplitude_state_ * sin(
            ((static_cast<double>(sample_num_)/static_cast<double>(SAMPLE_RATE)) * 2.0 * PI * frequency_state_) - start_phase_
        );
    }
};
} // namespace oscillator
} // namespace instrument
#endif // STRINGSOUNDS_LIBS_INSTRUMENT_STRING_OSCILLATOR_H_
