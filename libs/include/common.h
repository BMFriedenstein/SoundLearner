/*
 * common.h
 *
 *  Created on: 03 Jan 2019
 *      Author: brandon
 */

#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

#include <stdint.h>
/* Application exit codes */
#define EXIT_NORMAL 0
#define EXIT_BAD_ARGS 1
#define EXIT_READ_FILE_FAILED 2

/* Application constants */
#define SAMPLE_RATE 44100
#define BITDEPTH 16
#define MIN_AMP -32768
#define MAX_AMP 32767
#define PI 3.14159265


/* Wav format
 * Positions    Sample Value    Description
 * 1 - 4   "RIFF"  Marks the file as a riff file. Characters are each 1 byte long.
 * 5 - 8   File size (integer) Size of the overall file - 8 bytes, in bytes (32-bit integer). Typically, you'd fill this in after creation.
 * 9 -12   "WAVE"  File Type Header. For our purposes, it always equals "WAVE".
 * 13-16   "fmt "  Format chunk marker. Includes trailing null
 * 17-20   16  Length of format data as listed above
 * 21-22   1   Type of format (1 is PCM) - 2 byte integer
 * 23-24   2   Number of Channels - 2 byte integer
 * 25-28   44100   Sample Rate - 32 byte integer. Common values are 44100 (CD), 48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
 * 29-32   176400  (Sample Rate * BitsPerSample * Channels) / 8.
 * 33-34   4   (BitsPerSample * Channels) / 8.1 - 8 bit mono2 - 8 bit stereo/16 bit mono4 - 16 bit stereo
 * 35-36   16  Bits per sample
 * 37-40   "data"  "data" chunk header. Marks the beginning of the data section.
 * 41-44   File size (data)    Size of the data section.
 * Sample values are given above for a 16-bit stereo source.
 *
 */
typedef struct  WAV_HEADER{
    char            riff[4] = {'R','I','F','F'};            // RIFF Header      Magic header
    uint32_t        chunk_size;         // RIFF Chunk Size
    char            wave[4] = {'W','A','V','E'};            // WAVE Header
    char            format[4] = {'f','m','t',' '};;          // FMT header
    uint32_t        sub_chunk_1_size;   // Size of the fmt chunk
    uint16_t        audio_format;       // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        num_of_channels;    // Number of channels 1=Mono 2=Sterio
    uint32_t        sample_rate;        // Sampling Frequency in Hz
    uint32_t        bytes_per_second;   // bytes per second
    uint16_t        block_allign;       // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bit_depth;          // Number of bits per sample
    char            sub_chunk_2_id[4]={'d','a','t','a'};  // "data"  string
    uint32_t        sub_chunk_2_size;   // Sampled data length

}wav_header;

#endif /* COMMON_COMMON_H_ */
