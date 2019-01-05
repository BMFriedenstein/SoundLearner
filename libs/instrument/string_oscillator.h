/*
 * sound_string.h
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */

#ifndef STRING_OSCILLATOR_H_
#define STRING_OSCILLATOR_H_

#include <memory>
#include <string>
#include <math.h>

#include "../include/common.h"

class StringOscillatorC {
private:
    /* Sinusoid's start definition */
    double start_phase;
    double start_frequency_factor;
    double start_amplitude_factor;

    /* Signal modification definition */
    double sustain_factor;
    double amp_attack_delta = 0;
    double amp_decay_rate = 0;
    double freq_attack_delta= 0;
    double freq_decay_rate = 0;

    /* Signal State */
    double max_amp = 0;
    double max_freq = 0;
    double amp_state = 0;
    double freq_state = 0;
    double base_freq = 0;
    double velocity = 0;
    uint32_t sample_no = 0;

    inline double wave(){
        return amp_state * sin ( (2*(sample_no/SAMPLE_RATE)*PI*freq_state) - start_phase );
    }
public:
    StringOscillatorC(   double a_phase,
                    double a_freq_factor,
                    double a_amp_factor,
                    double a_sus_factor,
                    double a_amp_decay,
                    double a_amp_attack,
                    double a_freq_decay,
                    double a_freq_attack);

    void PrimeString( double freq, double velocity );
    double NextSample( bool sustain );
    uint32_t GetSampleNumber() { return sample_no; }
    std::string ToJson();
    std::unique_ptr<StringOscillatorC> TuneString( uint8_t amount );
    static std::unique_ptr<StringOscillatorC> CreateUntunedString();
};

#endif /* STRING_OSCILLATOR_H_ */
