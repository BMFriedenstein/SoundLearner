/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * common.h
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */

#ifndef INCLUDE_COMMON_H_
#define INCLUDE_COMMON_H_

#include <cstdint>

// Application exit codes
const uint32_t EXIT_NORMAL = 0;
const uint32_t EXIT_BAD_ARGS = 1;
const uint32_t EXIT_READ_FILE_FAILED = 2;

// Application constants
const uint32_t SAMPLE_RATE = 44100;
const uint32_t BITDEPTH = 16;
const int32_t MIN_AMP = -32768;
const int32_t MAX_AMP = 32767;
const double PI = 3.14159265;

// Wav format
// Positions    Sample Value    Description
// 1 - 4   "RIFF"  Marks the file as a riff file.
//         Characters are each 1 byte long.
// 5 - 8   File size (integer) Size of the overall file - 8 bytes,
//         in bytes (32-bit integer). Typically, you'd fill this in after creation.
// 9 -12   "WAVE"  File Type Header. For our purposes, it always equals "WAVE".
// 13-16   "fmt "  Format chunk marker. Includes trailing null
// 17-20   16  Length of format data as listed above
// 21-22   1   Type of format (1 is PCM) - 2 byte integer
// 23-24   2   Number of Channels - 2 byte integer
// 25-28   44100   Sample Rate - 32 byte integer. Common values are 44100 (CD),
//         48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
// 29-32   176400  (Sample Rate// BitsPerSample// Channels) / 8.
// 33-34   4   (BitsPerSample// Channels) / 8.1 - 8 bit mono2 -
//         8 bit stereo/16 bit mono4 - 16 bit stereo
// 35-36   16  Bits per sample
// 37-40   "data"  "data" chunk header. Marks the beginning of the data section.
// 41-44   File size (data)    Size of the data section.
// Sample values are given above for a 16-bit stereo source.
typedef struct WAV_HEADER {
  char riff[4] = { 'R', 'I', 'F', 'F' };
  uint32_t chunk_size;
  char wave[4] = { 'W', 'A', 'V', 'E' };
  char format[4] = { 'f', 'm', 't', ' ' };
  uint32_t sub_chunk_1_size;
  uint16_t audio_format;
  uint16_t num_of_channels;
  uint32_t sample_rate;
  uint32_t bytes_per_second;
  uint16_t block_allign;
  uint16_t bit_depth;
  char sub_chunk_2_id[4] = { 'd', 'a', 't', 'a' };
  uint32_t sub_chunk_2_size;

} wav_header;

#endif // INCLUDE_COMMON_H_
