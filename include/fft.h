/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * fft.h
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */

#pragma once
#ifndef INCLUDE_FFT_H_
#define INCLUDE_FFT_H_

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <numeric>
#include <vector>

#include "include/common.h"

namespace fft {
namespace detail {
inline constexpr std::size_t kMaxFftSize = 0xFFFFU;
inline constexpr uint8_t kReal = 0;
inline constexpr uint8_t kImag = 1;
inline constexpr double kSpectrogramDisplayMin = 50.0;
inline constexpr double kSpectrogramDisplayMax = 190.0;

inline std::array<fftw_complex, kMaxFftSize> fft_in_buffer;
inline std::array<fftw_complex, kMaxFftSize> fft_out_buffer;
inline std::array<double, kMaxFftSize> scratch_buffer = {};
} // namespace detail

enum class WindowTypes { NONE, HANN, FLAT_TOP, UNIFORM, FORCE, HAMMING, KAISER_BESSEL, EXPONENTIAL };

template <std::size_t A> static inline void WindowFunction(std::array<double, A> &in_out_data, WindowTypes window_function) {
  switch (window_function) {
  case WindowTypes::HANN: {
    auto i{0U};
    std::transform(in_out_data.begin(), in_out_data.end(), in_out_data.begin(),
                   [&i](auto in_val) { return in_val * std::pow(std::sin(i++ / (2 * M_PI / A)), 2); });
  } break;

  // TODO(brandon): Handle other window functions
  case WindowTypes::FLAT_TOP:
  case WindowTypes::UNIFORM:
  case WindowTypes::FORCE:
  case WindowTypes::HAMMING:
  case WindowTypes::KAISER_BESSEL:
  case WindowTypes::EXPONENTIAL:
  case WindowTypes::NONE:
  default:
    break;
  }
}

template <std::size_t A>
static inline void OneSidedFFT(std::array<double, A> &in_out_data, double freq, WindowTypes window_function = WindowTypes::NONE) {
  constexpr std::size_t in_fft_size = 2 * A;
  static_assert(in_fft_size < detail::kMaxFftSize);

  WindowFunction(in_out_data, window_function);

  // Prepare input/output buffers for fft.
  std::memset(detail::fft_in_buffer.data(), 0, sizeof(fftw_complex) * 2 * A);
  std::memset(detail::fft_out_buffer.data(), 0, sizeof(fftw_complex) * 2 * A);
  for (auto i{0U}; i < A; ++i) {
    detail::fft_in_buffer[i][detail::kReal] = in_out_data[i];
  }

  // Perform fft
  fftw_plan p = fftw_plan_dft_1d(static_cast<int>(in_fft_size), detail::fft_in_buffer.data(), detail::fft_out_buffer.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  // Calculate power fft and place on return buffer.
  for (auto i{0U}; i < A; ++i) {
    const double magnitude{std::hypot(detail::fft_out_buffer[i][detail::kReal], detail::fft_out_buffer[i][detail::kImag])};
    const double power_density{std::pow(magnitude, 2) / (2 * A * freq)};
    const double log_val{-10 * std::log10(power_density)};
    in_out_data[i] = log_val;
  }

  // Cleanup FFTW object.
  fftw_destroy_plan(p);
  fftw_cleanup();
}

namespace spectrogram {

using SpectrogramBuffer = std::vector<double>;

static inline double NormalizeSpectrogramValue(double value) {
  if (std::isnan(value)) {
    return 0.0;
  }

  const double normalized = (value - detail::kSpectrogramDisplayMin) / (detail::kSpectrogramDisplayMax - detail::kSpectrogramDisplayMin);
  return std::clamp(1.0 - normalized, 0.0, 1.0);
}

static inline double BufferAt(const SpectrogramBuffer &buffer, std::size_t width, std::size_t col, std::size_t row) {
  return buffer[row * width + col];
}

static inline double &BufferAt(SpectrogramBuffer &buffer, std::size_t width, std::size_t col, std::size_t row) {
  return buffer[row * width + col];
}

static inline void OneSidedFFT(std::vector<double> &in_out_data, double freq, WindowTypes window_function = WindowTypes::NONE) {
  const std::size_t input_size = in_out_data.size();
  const std::size_t fft_size = input_size * 2;

  if (window_function == WindowTypes::HANN) {
    for (std::size_t i = 0; i < input_size; ++i) {
      in_out_data[i] *= std::pow(std::sin(static_cast<double>(i) / (2 * M_PI / static_cast<double>(input_size))), 2);
    }
  }

  std::vector<fftw_complex> fft_in_buffer(fft_size);
  std::vector<fftw_complex> fft_out_buffer(fft_size);
  for (std::size_t i = 0; i < input_size; ++i) {
    fft_in_buffer[i][detail::kReal] = in_out_data[i];
  }

  fftw_plan p = fftw_plan_dft_1d(static_cast<int>(fft_size), fft_in_buffer.data(), fft_out_buffer.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  for (std::size_t i = 0; i < input_size; ++i) {
    const double magnitude{std::hypot(fft_out_buffer[i][detail::kReal], fft_out_buffer[i][detail::kImag])};
    const double power_density{std::pow(magnitude, 2) / (2 * input_size * freq)};
    in_out_data[i] = -10 * std::log10(power_density);
  }

  fftw_destroy_plan(p);
  fftw_cleanup();
}

static inline SpectrogramBuffer CreateSpectrogram(const std::vector<double> &source_signal, std::size_t resolution) {
  SpectrogramBuffer spectrogram(resolution * resolution);
  std::vector<double> fft_window(resolution);
  const std::size_t increment_size = source_signal.size() / resolution;

  std::size_t source_idx{0U};
  for (std::size_t x = 0; x < resolution; ++x) {
    const auto copy_size = std::min(resolution, source_signal.size() - source_idx);
    std::copy_n(source_signal.begin() + static_cast<std::ptrdiff_t>(source_idx), copy_size, fft_window.begin());
    std::fill(fft_window.begin() + static_cast<std::ptrdiff_t>(copy_size), fft_window.end(), 0.0);

    OneSidedFFT(fft_window, SAMPLE_RATE);

    for (std::size_t y = 0; y < resolution; ++y) {
      BufferAt(spectrogram, resolution, x, y) = NormalizeSpectrogramValue(fft_window[y]);
    }
    source_idx += increment_size;
  }

  return spectrogram;
}

static inline SpectrogramBuffer CreateMelSpectrogram(const std::vector<double> &source_signal, std::size_t resolution, std::size_t fft_size_multiplier = 25) {
  const std::size_t fft_bin_count = resolution * fft_size_multiplier;
  const double log_max = std::log2(static_cast<double>(fft_bin_count + 1));
  SpectrogramBuffer mel_spectrogram(resolution * resolution);
  std::vector<double> fft_window(fft_bin_count);
  const auto increment_size = source_signal.size() / resolution;

  std::size_t source_idx{0U};
  for (std::size_t x = 0; x < resolution; ++x) {
    const auto copy_size = std::min(fft_bin_count, source_signal.size() - source_idx);
    std::copy_n(source_signal.begin() + static_cast<std::ptrdiff_t>(source_idx), copy_size, fft_window.begin());
    std::fill(fft_window.begin() + static_cast<std::ptrdiff_t>(copy_size), fft_window.end(), 0.0);

    OneSidedFFT(fft_window, SAMPLE_RATE);

    for (std::size_t y = 0; y < resolution; ++y) {
      const double log_start = std::pow(2.0, (static_cast<double>(y) * log_max) / resolution) - 1.0;
      const double log_end = std::pow(2.0, (static_cast<double>(y + 1) * log_max) / resolution) - 1.0;
      const auto bin_start = std::min<std::size_t>(static_cast<std::size_t>(std::floor(log_start)), fft_bin_count - 1);
      const auto bin_end = std::min<std::size_t>(std::max<std::size_t>(static_cast<std::size_t>(std::ceil(log_end)), bin_start + 1), fft_bin_count);
      const auto bin_count = bin_end - bin_start;

      if (bin_count > 0) {
        const double sum_val = std::accumulate(fft_window.begin() + static_cast<std::ptrdiff_t>(bin_start), fft_window.begin() + static_cast<std::ptrdiff_t>(bin_end), 0.0);
        BufferAt(mel_spectrogram, resolution, x, y) = NormalizeSpectrogramValue(sum_val / static_cast<double>(bin_count));
      }
    }
    source_idx += increment_size;
  }

  return mel_spectrogram;
}

template <std::size_t X, std::size_t Y = X>
static inline void CreateSpectrogram(const std::vector<double> &source_signal, std::array<std::array<double, Y>, X> &out_spectogram) {
  static_assert((Y * 2) < detail::kMaxFftSize);

  const std::size_t increment_size = source_signal.size() / X;

  std::size_t source_idx{0U};
  for (auto window = out_spectogram.begin(); window != out_spectogram.end(); ++window) {
    const auto copy_size = std::min(X, source_signal.size() - source_idx);

    std::memcpy(detail::scratch_buffer.data(), &source_signal[source_idx], sizeof(double) * copy_size);
    std::fill(detail::scratch_buffer.begin() + copy_size, detail::scratch_buffer.begin() + X, 0.0);

    OneSidedFFT(*reinterpret_cast<std::array<double, X> *>(&detail::scratch_buffer), SAMPLE_RATE);

    // Normalize to values between 0.0 and 1.0
    std::transform(detail::scratch_buffer.begin(), detail::scratch_buffer.begin() + X, window->begin(),
                   [](const auto &val) { return NormalizeSpectrogramValue(val); });

    source_idx += increment_size;
  }
}

template <std::size_t X, std::size_t Y = X>
static inline std::array<std::array<double, Y>, X> CreateSpectrogram(const std::vector<double> &source_signal) {
  std::array<std::array<double, Y>, X> return_spectogram = {{}};
  CreateSpectrogram(source_signal, return_spectogram);
  return return_spectogram;
}

template <std::size_t X, std::size_t Q = X * 25>
static inline void CreateMelSpectrogram(const std::vector<double> &source_signal, std::array<std::array<double, X>, X> &mel_spectogram) {
  static_assert((Q * 2) < detail::kMaxFftSize);
  const double log_max = std::log2(static_cast<double>(Q + 1));

  auto source_idx{0U};
  const auto increment_size = source_signal.size() / X;

  for (auto x{0U}; x < X; ++x) {
    const auto copy_size = std::min(Q, source_signal.size() - source_idx);

    std::memcpy(detail::scratch_buffer.data(), &source_signal[source_idx], sizeof(double) * std::min(Q, source_signal.size() - source_idx));
    std::fill(detail::scratch_buffer.begin() + copy_size, detail::scratch_buffer.begin() + Q, 0.0);

    OneSidedFFT(*reinterpret_cast<std::array<double, Q> *>(&detail::scratch_buffer), SAMPLE_RATE);

    // Map the Q FFT bins into X logarithmically spaced frequency buckets.
    for (auto y{0U}; y < X; ++y) {
      const double log_start = std::pow(2.0, (static_cast<double>(y) * log_max) / X) - 1.0;
      const double log_end = std::pow(2.0, (static_cast<double>(y + 1) * log_max) / X) - 1.0;
      const auto bin_start = std::min<std::size_t>(static_cast<std::size_t>(std::floor(log_start)), Q - 1);
      const auto bin_end = std::min<std::size_t>(std::max<std::size_t>(static_cast<std::size_t>(std::ceil(log_end)), bin_start + 1), Q);
      const auto bin_count = bin_end - bin_start;

      if (bin_count > 0) {
        const double sum_val = std::accumulate(detail::scratch_buffer.begin() + bin_start, detail::scratch_buffer.begin() + bin_end, 0.0);
        mel_spectogram[x][y] = NormalizeSpectrogramValue(sum_val / static_cast<double>(bin_count));
      }
    }

    source_idx += increment_size;
  }
}

template <std::size_t X, std::size_t Q = X * 25>
static inline std::array<std::array<double, X>, X> CreateMelSpectrogram(const std::vector<double> &source_signal) {
  std::array<std::array<double, X>, X> mel_spectogram{{}};
  CreateMelSpectrogram<X, Q>(source_signal, mel_spectogram);
  return mel_spectogram;
}

} // namespace spectrogram
} // namespace fft

#endif /* INCLUDE_FFT_H_ */
