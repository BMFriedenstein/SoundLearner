/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * img.h
 *  Created on: 05 Jan 2019
 *      Author: Brandon
 */

#ifndef SRC_FFT_IMG_H_
#define SRC_FFT_IMG_H_

#include <vector>
#include <memory>
#include <string>

#include "shared/common.h"

namespace img {
namespace bmp {

struct FileHeader{
     uint16_t file_type=0x4D42;          // File type always BM which is 0x4D42
     uint32_t file_size=0;               // Size of the file (in bytes)
     uint16_t reserved1=0;               // Reserved, always 0
     uint16_t reserved2=0;               // Reserved, always 0
     uint32_t offset_data=0;             // Start position of pixel data (bytes from the beginning of the file)
} __attribute__ ((packed)) ;

struct BMPInfoHeader {
     uint32_t size{ 0 };                      // Size of this header (in bytes)
     uint32_t width{ 0 };                      // width of bitmap in pixels
     uint32_t height{ 0 };                     // width of bitmap in pixels
                                              //       (if positive, bottom-up, with origin in lower left corner)
                                              //       (if negative, top-down, with origin in upper left corner)
     uint16_t planes{ 1 };                    // No. of planes for the target device, this is always 1
     uint16_t bit_count{ 32 };                 // No. of bits per pixel
     uint32_t compression{ 0 };               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
     uint32_t size_image{ 0 };                // 0 - for uncompressed images
     uint32_t x_pixels_per_meter{ 0 };
     uint32_t y_pixels_per_meter{ 0 };
     uint32_t colors_used{ 0 };               // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
     uint32_t colors_important{ 0 };          // No. of colors used for displaying the bitmap. If 0 all colors are required
} __attribute__ ((packed));

struct BMPColorHeader {
     uint32_t red_mask{ 0x00ff0000 };         // Bit mask for the red channel
     uint32_t green_mask{ 0x0000ff00 };       // Bit mask for the green channel
     uint32_t blue_mask{ 0x000000ff };        // Bit mask for the blue channel
     uint32_t alpha_mask{ 0xff000000 };       // Bit mask for the alpha channel
     uint32_t color_space_type{ 0x73524742 }; // Default "sRGB" (0x73524742)
     uint32_t unused[16]{ 0 };                // Unused data for sRGB color space
} __attribute__ ((packed));

class BMPReaderC {
 public:
   BMPReaderC(const std::string & a_filename);
   std::string HeaderToString();
 private:
   std::string filename;
   wav_header header;
   std::vector<char> wav_data;
};

union RGBA {
   struct {
      uint8_t B;
      uint8_t G;
      uint8_t R;
      uint8_t A;
   } __attribute__ ((packed));
   uint32_t rgba;
};



class BMPWriterC {
 public:
  BMPWriterC(const std::vector<std::vector<uint32_t> >& a_data);
  BMPWriterC(const std::vector<std::vector<double> >& a_data);
  void Write(const std::string a_file_name);

 private:
  enum ColorScaleType {
     GREYSCALE =0
  };
  uint32_t ValToColor( double val, ColorScaleType type = GREYSCALE);
  unsigned char bmppad[3] = {0,0,0};
  FileHeader fileheader;
  BMPInfoHeader infoheader;
  BMPColorHeader colorheader;
  uint32_t W;
  uint32_t H;
  std::vector<uint32_t> img_data;
};
}
}  // namespace img
#endif   // SRC_FFT_IMG_H_
