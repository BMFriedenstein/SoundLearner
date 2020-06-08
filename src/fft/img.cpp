/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * img.cpp
 *  Created on: 05 Jan 2019
 *      Author: Brandon
 */

#include "fft/img.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
namespace img {
namespace bmp {

uint32_t BMPWriterC::ValToColor(
   double val, ColorScaleType type) {
   if (val > 1) { val = 1; }
   else if (val < 0) { val = 0; }
   switch (type) {
      case GREYSCALE:
         uint8_t val_u = static_cast<uint8_t>(std::round(val * 0xff));
         RGBA val_rgba;
         val_rgba.A = 0xff;
         val_rgba.R = val_u;
         val_rgba.G = val_u;
         val_rgba.B = val_u;
         return val_rgba.rgba;
      default:
         return 0xffffffff;
   }
}

BMPWriterC::BMPWriterC(const std::vector<std::vector<uint32_t> >& a_data)  {
   W = a_data.size();
   H = W > 0 ? a_data[0].size() : 0;
   img_data = std::vector<uint32_t>(W*H);
   for (size_t i = H - 1; i > 0; --i) {
      for (size_t j = W - 1; j > 0; --j) {
         img_data[i * W + j] = ValToColor(double(a_data[j][i]));
      }
   }
   fileheader.offset_data = sizeof(fileheader) + sizeof(infoheader);
   fileheader.file_size = fileheader.offset_data + img_data.size()*4 + sizeof(colorheader);
   infoheader.size = sizeof(infoheader);
   infoheader.width = W;
   infoheader.height = H;
   infoheader.compression = 0;
   infoheader.bit_count = 32;
}

BMPWriterC::BMPWriterC(const std::vector<std::vector<double> >& a_data) {
   W = a_data.size();
   H = W > 0 ? a_data[0].size() : 0;
   img_data = std::vector<uint32_t>(W * H);
   for (size_t i = W - 1; i > 0; --i) {
      for (size_t j = H - 1; j > 0; --j) {
         img_data[i * H + j] = ValToColor(double(a_data[j][i]));
      }
   }
   fileheader = FileHeader();
   fileheader.offset_data = sizeof(fileheader.offset_data) + sizeof(infoheader);
   fileheader.file_size = fileheader.offset_data + img_data.size()*4 + sizeof(colorheader);

   infoheader.size = sizeof(infoheader);
   infoheader.width = W;
   infoheader.height = H;
   infoheader.compression = 0;
   infoheader.bit_count = 32;
}

void BMPWriterC::Write(const std::string a_file_name) {
   if (W == 0 || H == 0) {
      std::cout << "WARNING, cannot write img as W || H is 0";
      return;
   }

   std::fstream fout;
   fout = std::fstream(a_file_name, std::ios::out | std::ios::binary);
   fout.write((const char*) &fileheader, sizeof(fileheader));
   fout.write((const char*) &infoheader, sizeof(infoheader));
   fout.write((const char*) img_data.data(), img_data.size() * 4);
   fout.write((const char*) &colorheader, sizeof(colorheader));
   fout.close();
}
}
}  // namespace img
