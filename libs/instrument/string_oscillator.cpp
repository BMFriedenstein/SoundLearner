/*
 * sound_string.cpp
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */

#include "string_oscillator.h"

#include <random>
using namespace std;

StringOscillatorC::StringOscillatorC( double a_phase,
                            double a_freq_factor,
                            double a_amp_factor,
                            double a_sus_factor,
                            double a_amp_decay,
                            double a_amp_attack,
                            double a_freq_decay,
                            double a_freq_attack) {

    // Restrictions
    if( a_amp_decay > 1 ) { a_amp_decay = 1; }
    if( a_amp_decay < 0 ) { a_amp_decay = 0; }
    if( a_freq_decay > 1 ) { a_amp_decay = 1; }
    if( a_freq_decay < 0 ) { a_amp_decay = 0; }
    if( a_amp_attack > a_amp_factor ) { a_amp_attack = a_amp_factor; }
    if( a_amp_attack < 0 ) { a_amp_attack = 0; }
    if( a_freq_attack > a_freq_factor ) { a_freq_attack = a_freq_factor; }
    if( a_freq_attack < 0 ) { a_freq_attack = 0; }
    if( a_freq_factor < 0 ) { a_freq_factor = 0; }
    if( a_freq_factor > 400 ) { a_freq_factor = 400; }
    if( a_amp_factor < 0 ) { a_amp_factor = 0; }
    if( sustain_factor < 0 ) { sustain_factor = 0; }

    start_phase = a_phase;
    start_frequency_factor = a_freq_factor;
    start_amplitude_factor = a_amp_factor;
    sustain_factor = a_sus_factor;
    amp_decay_rate = a_amp_decay;
    amp_attack_delta = a_amp_attack;
    freq_decay_rate = a_freq_decay;
    freq_attack_delta = a_freq_attack;
}

/*
 * Parse information required to generate a signal
 *
 * @parameters: frequency (The base note), velocity( how hard of note was pressed  form 0-1)
 * @returns: void
 */
void StringOscillatorC::PrimeString( double freq, double velocity ) {
    max_amp = velocity*start_amplitude_factor;
    max_freq = freq*start_frequency_factor;
    if( max_freq >SAMPLE_RATE/2 ) max_freq = SAMPLE_RATE/2;
    amp_state = 0;
    base_freq = freq;
    freq_state = base_freq;
    sample_no = 0;
}

/*
 * Generate the value of the next sample of the signal
 *
 * @parameters: sustain (Is the note still being pressed)
 * @returns: next value of the signal float
 */
double StringOscillatorC::NextSample( bool sustain ) {

    // Calculate amplitude state
    if( !in_amp_decay ){
        amp_state =  amp_state + amp_attack_delta;
    }
    else{
        double decay = (sustain ? sustain_factor : 1) * amp_decay_rate;
        amp_state = amp_state*(decay>1 ? 1 : decay) ;
    }
    if( amp_state <0 ){ amp_state = 0; }
    if( amp_state > max_amp ){
        in_amp_decay = true;
        amp_state = max_amp;
    }

    // Calculate frequency state
    if( !in_freq_decay ){
        freq_state = freq_state + freq_attack_delta;
    }
    else{
        double decay = (sustain ? sustain_factor : 1) * freq_decay_rate;
        freq_state = freq_state*(decay>1 ? 1 : decay) ;
    }
    if( freq_state <0 ){ freq_state =0; }
    if( freq_state > max_freq){
        in_freq_decay=true;
        freq_state = max_freq;
    }

    // generate sample;
    double sample_val = SineWave();

    // increment sample number
    sample_no++;

    return sample_val;
}

/*
 * Create a JSON representation of the SoundString.
 *
 * @parameters: none
 * @returns: json string
 */
string StringOscillatorC::ToJson() {
    string json_str = "{\n";
    json_str += "\"start_phase\":" +  to_string(start_phase) + ",\n";
    json_str += "\"start_frequency_factor\":" +  to_string(start_frequency_factor) + ",\n";
    json_str += "\"start_amplitude_factor\":" +  to_string(start_amplitude_factor) + ",\n";
    json_str += "\"sustain_factor\":" +  to_string(sustain_factor) + ",\n";
    json_str += "\"amp_decay_rate\":" +  to_string(amp_decay_rate) + ",\n";
    json_str += "\"amp_attack_delta\":" +  to_string(amp_attack_delta) + ",\n";
    json_str += "\"freq_decay_rate\":" +  to_string(freq_decay_rate) + ",\n";
    json_str += "\"freq_attack_delta\":" +  to_string(freq_attack_delta) + "\n";
    json_str += "}\n";
    return json_str;
}

/*
 * Returns a mutated version of the string, each parameter of the string sound only has a 50% likelihood of being mutated
 * @parameters: severity(determines the severity of the mutation)
 * @returns: Sound string pointer
 */
unique_ptr<StringOscillatorC> StringOscillatorC::TuneString(uint8_t severity) {
    float sev_factor = severity/255L;

    std::random_device random_device; // obtain a random number from hardware
    std::mt19937 eng(random_device()); // seed the generator
    std::uniform_real_distribution<> real_distr(-sev_factor, sev_factor); // define the range
    double a_phase =  start_phase + (real_distr(eng) > 0) ? 2*PI*real_distr(eng) : 0;
    double a_freq_factor = start_frequency_factor + (real_distr(eng) > 0) ? 10*real_distr(eng) : 0;
    double a_amp_factor = start_amplitude_factor + (real_distr(eng) > 0) ? 30000*real_distr(eng) : 0;
    double a_sus_factor = sustain_factor + (real_distr(eng) > 0) ? 0.2*real_distr(eng) : 0;
    double a_amp_decay = amp_decay_rate + (real_distr(eng) > 0) ? 0.1*real_distr(eng) : 0;
    double a_amp_attack = amp_attack_delta + (real_distr(eng) > 0) ? 500*real_distr(eng) : 0;
    double a_freq_decay = freq_decay_rate + (real_distr(eng) > 0) ? 0.1*real_distr(eng) : 0;
    double a_freq_attack = freq_attack_delta + (real_distr(eng) > 0) ? real_distr(eng) : 0;

    auto mutant_string = unique_ptr<StringOscillatorC> { new StringOscillatorC(a_phase,
            a_freq_factor, a_amp_factor, a_sus_factor, a_amp_decay,
            a_amp_attack, a_freq_decay, a_freq_attack) };

    return mutant_string;
}

/*
 * Generates a new completely randomized soundstring
 * @parameters: none
 * @returns: Sound string pointer
 */
unique_ptr<StringOscillatorC> StringOscillatorC::CreateUntunedString() {
    std::random_device random_device; // obtain a random number from hardware
    std::mt19937 eng(random_device()); // seed the generator
    std::uniform_real_distribution<> real_distr(0, 1); // define the range
    double a_phase = 2*PI*real_distr(eng);
    double a_freq_factor = 10*real_distr(eng);
    double a_amp_factor = 30000*real_distr(eng);
    double a_sus_factor = 0.99+ 0.02*real_distr(eng);
    double a_amp_decay = 0.965 + 0.035*real_distr(eng);
    double a_amp_attack = 500*real_distr(eng);
    double a_freq_decay =  0.999 + 0.001*real_distr(eng);
    double a_freq_attack = 20*real_distr(eng);

    auto mutant_string = unique_ptr<StringOscillatorC> { new StringOscillatorC(a_phase,
            a_freq_factor, a_amp_factor, a_sus_factor, a_amp_decay,
            a_amp_attack, a_freq_decay, a_freq_attack) };

    return mutant_string;
}
