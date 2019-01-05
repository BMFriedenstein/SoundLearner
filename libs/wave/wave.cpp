/*
 * wave.cpp
 *
 *  Created on: 05 Jan 2019
 *      Author: brandon
 */

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "wave.h"
using namespace std;

WaveReaderC::WaveReaderC( string & a_filename ){
    filename= a_filename;

    // Open file
    ifstream wav_file(a_filename, ios::binary);
    if ( !wav_file.is_open() ) {
        cout << "ERROR!!! Could not open wave file " << a_filename << endl;
        exit(EXIT_READ_FILE_FAILED);
    }
    wav_file.unsetf(std::ios::skipws);

    // Get file size
    wav_file.seekg(0, ios::end);
    size_t file_size  = wav_file.tellg();
    wav_file.seekg(0, std::ios::beg);

    if (file_size <= sizeof(wav_header)) {
       cout << "ERROR!!! Could not open wave file " << a_filename << " file size to small " << endl;
       exit(EXIT_READ_FILE_FAILED);
    }

    // Read header and data
    vector<char> buffer(file_size);
    cout << "DEBUG : header_size: " << sizeof(wav_header) << endl;
    wav_file.read(buffer.data(), file_size);
    memcpy( &header, buffer.data(), sizeof(wav_header));
    wav_data = std::vector<char>(buffer.begin() + sizeof(wav_header), buffer.end());
}

string WaveReaderC::HeaderToString(){
    string ret_string = "{\n";
    ret_string += "\"riff\": "  + string(header.riff,4) + ",\n";
    ret_string += "\"chunk_size\": "  + to_string(header.chunk_size) + ",\n";
    ret_string += "\"wave\": "  + string(header.wave,4) + ",\n";
    ret_string += "\"format\": "  + string(header.format,4) + ",\n";
    ret_string += "\"sub_chunk_1_size\": "  + to_string(header.sub_chunk_1_size) + ",\n";
    ret_string += "\"audio_format\": "  + to_string(header.audio_format) + ",\n";
    ret_string += "\"num_of_channels\": "  + to_string(header.num_of_channels) + ",\n";
    ret_string += "\"sample_rate\": "  + to_string(header.sample_rate) + ",\n";
    ret_string += "\"bytes_per_second\": "  + to_string(header.bytes_per_second) + ",\n";
    ret_string += "\"block_allign\": "  + to_string(header.block_allign) + ",\n";
    ret_string += "\"bit_depth\": "  + to_string(header.bit_depth) + ",\n";
    ret_string += "\"sub_chunk_2_id\": "  + string(header.sub_chunk_2_id,4) + ",\n";
    ret_string += "\"sub_chunk_2_size\": "  + to_string(header.sub_chunk_2_size) + ",\n";
    ret_string += "\"data_size\": "  + to_string(wav_data.size()) + "\n";
    ret_string += "}\n";
    return ret_string;
}


vector<int16_t> WaveReaderC::ToMono16BitWave(){
    // Mono
    if( header.num_of_channels == 1 ){
        // 16 bit
        if( header.bit_depth == 16 ){
            return vector<int16_t>((int16_t*)wav_data.begin(), (int16_t*)wav_data.end());
        }
        // TODO: Handle other Bit depth
    }

    // TODO: Handle other formats
    return vector<int16_t>(wav_data.size());
}
