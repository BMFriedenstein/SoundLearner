/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * structures.h
 *  Created on: 03 June 2020
 *      Author: Brandon
 */
#pragma once
#ifndef INCLUDE_STRUCTURES_H_
#define INCLUDE_STRUCTURES_H_

/** Wav format
      Positions    Sample Value    Description
      1 - 4   "RIFF"  Marks the file as a riff file.
              Characters are each 1 byte long.
      5 - 8   File size (integer) Size of the overall file - 8 bytes,
              in bytes (32-bit integer). Typically, you'd fill this in after
              creation.
      9 -12   "WAVE"  File Type Header. For our purposes, it always equals "WAVE".
      13-16   "fmt "  Format chunk marker. Includes trailing null
      17-20   16  Length of format data as listed above
      21-22   1   Type of format (1 is PCM) - 2 byte integer
      23-24   2   Number of Channels - 2 byte integer
      25-28   44100   Sample Rate - 32 byte integer. Common values are 44100 (CD),
              48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
      29-32   176400  (Sample Rate// BitsPerSample// Channels) / 8.
      33-34   4   (BitsPerSample// Channels) / 8.1 - 8 bit mono2 -
              8 bit stereo/16 bit mono4 - 16 bit stereo
      35-36   16  Bits per sample
      37-40   "data"  "data" chunk header. Marks the beginning of the data section.
      41-44   File size (data)    Size of the data section.
      Sample values are given above for a 16-bit stereo source.
*/
struct WavFileHeader {
  char riff[4] = {'R', 'I', 'F', 'F'};
  uint32_t chunk_size;
  char wave[4] = {'W', 'A', 'V', 'E'};
  char format[4] = {'f', 'm', 't', ' '};
  uint32_t sub_chunk_1_size;
  uint16_t audio_format;
  uint16_t num_of_channels;
  uint32_t sample_rate;
  uint32_t bytes_per_second;
  uint16_t block_allign;
  uint16_t bit_depth;
  char sub_chunk_2_id[4] = {'d', 'a', 't', 'a'};
  uint32_t sub_chunk_2_size;
} __attribute__((packed));

union RGBA {
  struct RSGBSt {
    uint8_t B;
    uint8_t G;
    uint8_t R;
    uint8_t A;
  } __attribute__((packed)) rgba_st;
  uint32_t rgba;
};

struct BMPFileHeader {
  uint16_t file_type = 0x4D42;  // File type always BM which is 0x4D42
  uint32_t file_size = 0;       // Size of the file (in bytes)
  uint16_t reserved1 = 0;       // Reserved, always 0
  uint16_t reserved2 = 0;       // Reserved, always 0
  uint32_t offset_data = 0;     // Start position of pixel data (bytes from the beginning of the file)
} __attribute__((packed));

struct BMPInfoHeader {
  uint32_t size{0};         // Size of this header (in bytes)
  uint32_t width{0};        // width of bitmap in pixels
  uint32_t height{0};       // width of bitmap in pixels
                            //       (if positive, bottom-up, with origin in lower left corner)
                            //       (if negative, top-down, with origin in upper left corner)
  uint16_t planes{1};       // No. of planes for the target device, this is always 1
  uint16_t bit_count{32};   // No. of bits per pixel
  uint32_t compression{0};  // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY
                            // UNCOMPRESSED BMP images
  uint32_t size_image{0};   // 0 - for uncompressed images
  uint32_t x_pixels_per_meter{0};
  uint32_t y_pixels_per_meter{0};
  uint32_t colors_used{0};       // No. color indexes in the color table. Use 0 for
                                 // the max number of colors allowed by bit_count
  uint32_t colors_important{0};  // No. of colors used for displaying the bitmap.
                                 // If 0 all colors are required
} __attribute__((packed));

struct BMPColorHeader {
  uint32_t red_mask{0x00ff0000};          // Bit mask for the red channel
  uint32_t green_mask{0x0000ff00};        // Bit mask for the green channel
  uint32_t blue_mask{0x000000ff};         // Bit mask for the blue channel
  uint32_t alpha_mask{0xff000000};        // Bit mask for the alpha channel
  uint32_t color_space_type{0x73524742};  // Default "sRGB" (0x73524742)
  uint32_t unused[16]{0};                 // Unused data for sRGB color space
} __attribute__((packed));

#endif  // INCLUDE_STRUCTURES_H_
