/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * fft.cpp
 *  Created on: 05 June 2019
 *      Author: Brandon
 */


#include "fft.h"
#include <complex>
#include <fftw3.h>
#include <iostream>
namespace Fft {
void WindowFunction(std::vector<double>& in_out_data, WindowTypes window_function) {
   switch(window_function) {
      case FLAT_TOP:
      case UNIFORM:
      case FORCE:
      case HAMMING:
      case KAISER_BESSEL:
      case EXPONENTIAL:
      case NONE:
         break;
      case HANN:
         const double scale_val = PI / in_out_data.size();
         for (size_t i = 0; i < in_out_data.size(); ++i) {
            in_out_data[i] *= std::pow(std::sin(i / scale_val), 2);
         }
         break;
      default:
         break;
   }
}

void OneSidedFFT(std::vector<double>& in_out_data, uint32_t fft_size, const double freq, WindowTypes window_function){
   const uint32_t in_fft_size = 2 * fft_size;
   Fft::WindowFunction(in_out_data, window_function);
   in_out_data.resize(in_fft_size);
   fftw_complex in[in_fft_size];
   fftw_complex out[in_fft_size];
   for (size_t i=0; i<in_fft_size; ++i) {
      in[i][real] = in_out_data[i];
      in[i][imag] = 0;
   }

   fftw_plan p = fftw_plan_dft_1d(in_fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
   fftw_execute(p);

   in_out_data.resize(fft_size);
   for (auto i = 0; i < fft_size; ++i) {
      double magnitude { std::abs(std::complex<double>(out[i][real], out[i][imag])) };
      double power_density { std::pow(magnitude, 2) / (2 * fft_size * freq) };
      double log_val { 10 * std::log10(power_density) };
      in_out_data[i] = log_val;
   }
   fftw_destroy_plan(p);
   fftw_cleanup();
}



namespace spectrogram {

std::vector<double> LogTransform(const uint32_t tgt_resolution, std::vector<std::complex<double> >& vec) {
   return std::vector<double>();
}
}
}
