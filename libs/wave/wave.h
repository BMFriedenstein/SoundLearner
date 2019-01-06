/*
 * wave.h
 *
 *  Created on: 05 Jan 2019
 *      Author: brandon
 */

#ifndef LIBS_WAVE_WAVE_H_
#define LIBS_WAVE_WAVE_H_

#include <vector>
#include <memory>
#include <string>
#include "include/common.h"

class WaveReaderC {
private:
    std::string filename;
    wav_header header;
    std::vector<char> wav_data;
public:

    WaveReaderC(std::string & a_filename);
    std::string HeaderToString();
    std::vector<int16_t> ToMono16BitWave();


    // Todo for future
    std::vector<int32_t> ToMono32BitWave();
    std::vector<float> ToMonoFloatWave();
};

class MonoWaveWriterC {
private:
    wav_header header;
    std::vector<char> wav_data;
public:

    MonoWaveWriterC( std::vector<int16_t>& a_data );
    void Write( std::string a_file_name );
};



#endif /* LIBS_WAVE_WAVE_H_ */
