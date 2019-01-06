/*
 * model.cpp
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */
#include "instrument_model.h"

#include <stdio.h>
#include <iostream>
using namespace std;

InstrumentModelC::InstrumentModelC( uint16_t a_num_strings, string a_instrument_name ){
    name = a_instrument_name;
    for( uint16_t i =0; i < a_num_strings; i++ ){
        AddUntunedString();
    }
}

/*
 * Add a pre tuned string to the instrument
 * @parameters: reference to a a_tuned_string (StringOscillatorC),
 * @returns none
 */
void InstrumentModelC::AddTunedString(StringOscillatorC& a_tuned_string){
    unique_ptr<StringOscillatorC> tuned_string( new StringOscillatorC( move(a_tuned_string) ) );
    sound_strings.push_back(std::move(tuned_string));
}

/*
 * Add a randomly tuned string sound to the instrument
 * @parameters: none,
 * @returns none
 */
void InstrumentModelC::AddUntunedString(){
    sound_strings.push_back(std::move(StringOscillatorC::CreateUntunedString()));
}

/*
 * Create a JSON representation of the instrument
 * @parameters: none,
 * @returns JSON string
 */
string InstrumentModelC::ToJson(){
    string return_json = "{\n";
    return_json += "\"name\": \"" + name + "\",\n";
    return_json += "\"strings\": {\n";
    for( auto string_iter = begin (sound_strings); string_iter != end (sound_strings); ++string_iter ){
        return_json += string_iter->get()->ToJson();
        if ( string_iter +1 != sound_strings.end()){
            return_json += ",\n";
        }
    }
    return_json += "}\n}\n";
    return return_json;

}

/*
 * Generates a array of double sample values representing the sound of the note played
 * @parameters: velocity(speed of note played), frequency(Which note), number of sample to generate, array of sustain values
 * @returns: vector of doubles
 */
vector<double> InstrumentModelC::GenerateSignal( double velocity, double frequency, uint32_t num_of_samples, vector<bool>& sustain ){
    vector<double> Signal(num_of_samples);

    // Check that we have a sustain value for each sample
    if( sustain.size() != num_of_samples ){
        cout << "Warning!!! Sustain array not equal to sample length" << endl;
        sustain.resize(num_of_samples);
    }

    // Initiate each of the strings
    for( auto string_iter = begin (sound_strings); string_iter != end (sound_strings); ++string_iter ){
        string_iter->get()->PrimeString(frequency,velocity);
    }

    // Generate samples
    for( uint32_t i =0; i < num_of_samples; i++ ){
        double sample_val = 0;

        for( auto string_iter = begin (sound_strings); string_iter != end (sound_strings); ++string_iter ){
            sample_val += string_iter->get()->NextSample(sustain[i]);
        }
        Signal[i] = sample_val;
    }

    return Signal;
}

/*
 * Generates a array of rounded integer sample values representing the sound of the note played
 * @parameters: velocity(speed of note played), frequency(Which note), number of sample to generate, array of sustain values
 * @returns: vector of integers
 */
vector<int16_t> InstrumentModelC::GenerateIntSignal( double velocity, double frequency, uint32_t num_of_samples, vector<bool>& sustain ){
    vector<int16_t> Signal(num_of_samples);

    // Check that we have a sustain value for each sample
    if( sustain.size() != num_of_samples){
        cout << "Warning!!! Sustain array not equal to sample length" << endl;
        sustain.resize(num_of_samples);
    }

    // Initiate each of the strings
    for( auto string_iter = begin (sound_strings); string_iter != end (sound_strings); ++string_iter ){
        string_iter->get()->PrimeString( frequency, velocity );
    }

    // Generate samples
    for( uint32_t i =0; i < num_of_samples; i++ ){
        double sample_val = 0;
        for( auto string_iter = begin (sound_strings); string_iter != end (sound_strings); ++string_iter ){
            sample_val += string_iter->get()->NextSample( sustain[i] );
        }

        if( sample_val > MAX_AMP){ sample_val = MAX_AMP; }
        if( sample_val < MIN_AMP){ sample_val = MIN_AMP; }

        if(sample_val <0){
            Signal[i] = (int16_t)int(sample_val - 0.5);
        }
        else{
            Signal[i] = (int16_t)int(sample_val + 0.5);
        }
    }

    return Signal;
}
