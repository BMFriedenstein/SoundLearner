/*
 * model.h
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */

#ifndef TRAINER_MODEL_H_
#define TRAINER_MODEL_H_

#include <memory>
#include <vector>
#include "sound_string.h"

class InstrumentModelC {
private:
    const uint16_t max_strings = 1000;
	std::vector<std::unique_ptr<SoundStringC>> sound_strings;
	std::string name;
public:
	InstrumentModelC( uint16_t a_num_strings, std::string a_instrument_name );
	std::string GetName(){ return name; };

	void AddTunedString(SoundStringC& a_tuned_string);
	void AddUntunedString();

	std::string ToJson();
	std::vector<double> GenerateSignal( double velocity, double frequency, uint32_t num_of_samples , std::vector<bool>& sustain);
	std::vector<int16_t> GenerateIntSignal( double velocity, double frequency, uint32_t num_of_samples, std::vector<bool>& sustain );

	// TODO for player
	void PrimeNotePlayed( double frequency, double velocity);
	double GenerateNextSample( bool sustain );
};


#endif /* TRAINER_MODEL_H_ */
